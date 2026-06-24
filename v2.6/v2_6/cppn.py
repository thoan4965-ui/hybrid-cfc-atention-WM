"""CPPN: policy + prediction + modular (8 modules) + dopamine."""
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

@jit
def genome_to_policy(nodes, conns):
    """Modular: all nodes visible, only connections per module."""
    has_mod = nodes.shape[-1] >= 8
    w_ih = jnp.zeros((30, 10)); w_ho = jnp.zeros((10, 8))
    w_pred = jnp.zeros((10, 29)); w_dopa = jnp.zeros(3)
    base_nodes = nodes[..., :7]
    for mod in range(8):
        if has_mod:
            mask_c = ~jnp.isnan(conns[:, 0]) & (conns[:, 6] == mod) & (conns[:, 4] > 0.5)
            mc = jnp.where(mask_c[:, None], conns, jnp.nan)
        else:
            mc = conns
        w_ih += cppn_query(base_nodes, mc, SI).reshape(30, 10)
        w_ho += cppn_query(base_nodes, mc, SH).reshape(10, 8)
        w_pred += cppn_query(base_nodes, mc, SP).reshape(10, 29)
        w_dopa += cppn_query(base_nodes, mc, SD)
    return {'w_ih': w_ih / 8, 'w_ho': w_ho / 8, 'w_pred': w_pred / 8, 'w_dopa': w_dopa / 8}

def policy_forward(params, obs):
    h = jnp.tanh(obs @ params['w_ih'][:-1] + params['w_ih'][-1])
    action = jnp.tanh(h @ params['w_ho'])
    pred_next = h @ params['w_pred']
    return action, pred_next
