"""CPPN: policy + prediction + modular (8 modules) + dopamine + regulatory + spatial + thought projections."""
import jax, jax.numpy as jnp
from jax import lax, vmap, jit
from v2_6.genome import MAX_GENES

def cppn_query(nodes, conns, coords):
    n_nodes = jnp.nan_to_num(nodes, 0); n_conns = jnp.nan_to_num(conns, 0)
    is_node = ~jnp.isnan(nodes[:, 0]); is_conn = ~jnp.isnan(conns[:, 0]) & (conns[:, 4] > 0.5)
    def eval_one(coord):
        v = jnp.zeros(MAX_GENES)
        for ci in range(4): v = v.at[ci].set(coord[ci] + jnp.where(is_node[ci], nodes[ci,5], 0.))
        def sf(vc, ci):
            v,_=vc; has=is_conn[ci]
            si=jnp.argmax(is_node*(jnp.abs(nodes[:,0]-conns[ci,1])<0.01))
            di=jnp.argmax(is_node*(jnp.abs(nodes[:,0]-conns[ci,2])<0.01))
            update = jnp.where(has, conns[ci,3]*jnp.tanh(v[si])+nodes[di,5], 0.)
            v=v.at[di].add(update)
            return(v,ci),None
        (v,_),_=lax.scan(sf,(v,0),jnp.arange(MAX_GENES))
        last=jnp.clip(jnp.sum(is_node)-1,0).astype(jnp.int32)
        return jnp.tanh(v[last])
    return vmap(eval_one)(coords)

def _sub():
    no=30; nh=10; na=8; np_obs=29; n_spat=46; n_thought=16
    SIH = no * nh; SPAT = 46 * nh; THOUGHT = 16 * nh  # spatial + thought projections
    coords = jnp.zeros((SIH + SPAT + THOUGHT + nh*na + nh*np_obs + 3, 4))
    idx = 0
    # ih (obs → hidden) 30×10 = 300
    for i in range(no):
        for h in range(nh):
            coords = coords.at[idx].set(jnp.array([i/no*2-1, -1., h/nh*2-1, 0.]))
            idx += 1
    # spatial_proj (46 → 10) = 460
    for s in range(n_spat):
        for h in range(nh):
            coords = coords.at[idx].set(jnp.array([s/n_spat*2-1, -2., h/nh*2-1, 3.]))
            idx += 1
    # thought_proj (16 → 10) = 160
    for t in range(n_thought):
        for h in range(nh):
            coords = coords.at[idx].set(jnp.array([t/n_thought*2-1, -3., h/nh*2-1, 4.]))
            idx += 1
    # ho (hidden → action) 10×8 = 80
    for h in range(nh):
        for o in range(na):
            coords = coords.at[idx].set(jnp.array([h/nh*2-1, 0., o/na*2-1, 1.]))
            idx += 1
    # pred (hidden → pred_obs) 10×29 = 290
    for h in range(nh):
        for p in range(np_obs):
            coords = coords.at[idx].set(jnp.array([h/nh*2-1, 1., p/np_obs*2-1, 2.]))
            idx += 1
    # dopa 3
    for d in range(3):
        coords = coords.at[idx].set(jnp.array([d/3*2-1, -2., 0., 3.]))
        idx += 1
    return coords, SIH, SPAT, THOUGHT

COORDS, SI_H, SI_SPAT, SI_THOUGHT = _sub()
SI_H_END = SI_H
SI_SPAT_END = SI_H + SI_SPAT
SI_THOUGHT_END = SI_H + SI_SPAT + SI_THOUGHT

def grid_encoding(x, y, scale=1.0):
    ang = jnp.array([0., jnp.pi/3, 2*jnp.pi/3])
    phases = jnp.linspace(0, 2*jnp.pi, 10)
    ph = x * jnp.cos(ang[:, None]) * scale + y * jnp.sin(ang[:, None]) * scale
    return jnp.sin(ph + phases[None, :]).ravel()

def place_encoding(x, y, n_places=16):
    xs = jnp.linspace(-20., 20., 4)
    ys = jnp.linspace(-20., 20., 4)
    gx, gy = jnp.meshgrid(xs, ys)
    d = (x - gx.ravel())**2 + (y - gy.ravel())**2
    return jnp.exp(-d / (2 * 4.**2))

def spatial_encoding(x, y, scale=1.0):
    return jnp.concatenate([grid_encoding(x, y, scale), place_encoding(x, y)])

@jit
def genome_to_policy(nodes, conns, regs=None):
    """Modular CPPN with support for spatial + thought projections."""
    # Handle both 2D (per-agent) and 3D (batched) inputs
    if nodes.ndim == 3:
        nodes = nodes[0]
        conns = conns[0]

    has_mod = nodes.shape[-1] >= 8
    n_ih = 30 * 10; n_spat = 46 * 10; n_thought = 16 * 10
    w_ih = jnp.zeros((30, 10)); w_ho = jnp.zeros((10, 8))
    w_pred = jnp.zeros((10, 29)); w_dopa = jnp.zeros(3)
    w_spat = jnp.zeros((46, 10)); w_thought = jnp.zeros((16, 10))
    base_nodes = nodes[..., :7]

    if regs is not None:
        module_on = jax.nn.sigmoid(regs[:8]) > 0.5
        n_active = jnp.maximum(jnp.sum(module_on), 1)
    else:
        module_on = jnp.ones(8, dtype=jnp.bool_)
        n_active = 8

    for mod in range(8):
        if has_mod:
            mask_c = ~jnp.isnan(conns[:, 0]) & (conns[:, 6] == mod) & (conns[:, 4] > 0.5)
            mc = jnp.where(mask_c[:, None], conns, jnp.nan)
        else:
            mc = conns
        mod_on = module_on[mod]
        out = cppn_query(base_nodes, mc, COORDS)
        w_ih += mod_on * out[:SI_H_END].reshape(30, 10)
        w_spat += mod_on * out[SI_H_END:SI_SPAT_END].reshape(46, 10)
        w_thought += mod_on * out[SI_SPAT_END:SI_THOUGHT_END].reshape(16, 10)
        w_ho += mod_on * out[SI_THOUGHT_END:SI_THOUGHT_END + 80].reshape(10, 8)
        w_pred += mod_on * out[SI_THOUGHT_END + 80:SI_THOUGHT_END + 80 + 290].reshape(10, 29)
        w_dopa += mod_on * out[SI_THOUGHT_END + 80 + 290:].reshape(3)
    return {'w_ih': w_ih / n_active, 'w_ho': w_ho / n_active,
            'w_pred': w_pred / n_active, 'w_dopa': w_dopa / n_active,
            'w_spat': w_spat / n_active, 'w_thought': w_thought / n_active}

@jit
def policy_forward(params, obs, spat_enc=None, thought_state=None):
    h = jnp.tanh(obs @ params['w_ih'][:-1] + params['w_ih'][-1])
    if spat_enc is not None:
        h = h + spat_enc @ params['w_spat']
    if thought_state is not None:
        h = h + thought_state @ params['w_thought']
    h = jnp.tanh(h)
    action = jnp.tanh(h @ params['w_ho'])
    pred_next = h @ params['w_pred']
    return action, pred_next
