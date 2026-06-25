"""CPPN: policy + prediction + modular (8 modules) + dopamine + regulatory + spatial encoding."""
import jax, jax.numpy as jnp
from jax import lax, vmap, jit

def cppn_query(nodes, conns, coords):
    n_nodes = jnp.nan_to_num(nodes, 0); n_conns = jnp.nan_to_num(conns, 0)
    is_node = ~jnp.isnan(nodes[:, 0]); is_conn = ~jnp.isnan(conns[:, 0]) & (conns[:, 4] > 0.5)
    def eval_one(coord):
        v = jnp.zeros(100)
        for ci in range(4): v = v.at[ci].set(coord[ci] + jnp.where(is_node[ci], nodes[ci,5], 0.))
        def sf(vc, ci):
            v,_=vc; has=is_conn[ci]
            si=jnp.argmax(is_node*(jnp.abs(nodes[:,0]-conns[ci,1])<0.01))
            di=jnp.argmax(is_node*(jnp.abs(nodes[:,0]-conns[ci,2])<0.01))
            update = jnp.where(has, conns[ci,3]*jnp.tanh(v[si])+nodes[di,5], 0.)
            v=v.at[di].add(update)
            return(v,ci),None
        (v,_),_=lax.scan(sf,(v,0),jnp.arange(100))
        last=jnp.clip(jnp.sum(is_node)-1,0).astype(jnp.int32)
        return jnp.tanh(v[last])
    return vmap(eval_one)(coords)

def _sub():
    no=30; nh=10; na=8; np_obs=29
    ih=jnp.zeros((no*nh,4)); ho=jnp.zeros((nh*na,4))
    for i in range(no):
        for h in range(nh): ih=ih.at[i*nh+h].set(jnp.array([i/no*2-1,-1.,h/nh*2-1,0.]))
    for h in range(nh):
        for o in range(na): ho=ho.at[h*na+o].set(jnp.array([h/nh*2-1,0.,o/na*2-1,1.]))
    pred=jnp.zeros((nh*np_obs,4))
    for h in range(nh):
        for p in range(np_obs): pred=pred.at[h*np_obs+p].set(jnp.array([h/nh*2-1,1.,p/np_obs*2-1,2.]))
    dopa=jnp.zeros((3,4))
    for d in range(3): dopa=dopa.at[d].set(jnp.array([d/3*2-1,-2.,0.,3.]))
    return ih, ho, pred, dopa

SI, SH, SP, SD = _sub()

def grid_encoding(x, y, scale=1.0):
    """Grid cell encoding from (x,y). 3 orientations × 10 phases = 30 dim."""
    ang = jnp.array([0., jnp.pi/3, 2*jnp.pi/3])[:, None]
    ph_x = x * jnp.cos(ang) * scale
    ph_y = y * jnp.sin(ang) * scale
    return jnp.sin(ph_x + ph_y).ravel()

def place_encoding(x, y, n_places=16):
    """Simple place cell encoding. 16 fixed grid positions."""
    xs = jnp.linspace(-20., 20., 4)
    ys = jnp.linspace(-20., 20., 4)
    gx, gy = jnp.meshgrid(xs, ys)
    d = (x - gx.ravel())**2 + (y - gy.ravel())**2
    return jnp.exp(-d / (2 * 4.**2))

def spatial_encoding(x, y, scale=1.0):
    """Combined grid + place encoding: 30 + 16 = 46 dim."""
    return jnp.concatenate([grid_encoding(x, y, scale),
                            place_encoding(x, y)])

@jit
def genome_to_policy(nodes, conns, regs=None):
    """Modular: all nodes visible, only connections per module.
       regs (16,): first 8 = module enable bits (sigmoid > 0.5 = on).
       JIT-safe: uses jnp.where for gating, no Python if on traced arrays."""
    has_mod = nodes.shape[-1] >= 8
    w_ih = jnp.zeros((30, 10)); w_ho = jnp.zeros((10, 8))
    w_pred = jnp.zeros((10, 29)); w_dopa = jnp.zeros(3)
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
        w_ih += mod_on * cppn_query(base_nodes, mc, SI).reshape(30, 10)
        w_ho += mod_on * cppn_query(base_nodes, mc, SH).reshape(10, 8)
        w_pred += mod_on * cppn_query(base_nodes, mc, SP).reshape(10, 29)
        w_dopa += mod_on * cppn_query(base_nodes, mc, SD)
    return {'w_ih': w_ih / n_active, 'w_ho': w_ho / n_active,
            'w_pred': w_pred / n_active, 'w_dopa': w_dopa / n_active}

def policy_forward(params, obs):
    h = jnp.tanh(obs @ params['w_ih'][:-1] + params['w_ih'][-1])
    action = jnp.tanh(h @ params['w_ho'])
    pred_next = h @ params['w_pred']
    return action, pred_next
