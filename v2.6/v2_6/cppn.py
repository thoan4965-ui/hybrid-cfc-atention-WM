"""Coordinate CPPN — JIT-compilable with lax.fori_loop."""
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit

def cppn_query(nodes, conns, coords):
    """coord (N,4) → weights (N,) via CPPN. JIT-compatible."""
    n_nodes = jnp.nan_to_num(nodes, 0)
    n_conns = jnp.nan_to_num(conns, 0)
    is_node = ~jnp.isnan(nodes[:, 0])
    is_conn = ~jnp.isnan(conns[:, 0]) & (conns[:, 4] > 0.5)
    
    def eval_one(coord):
        v = jnp.zeros(100)
        v = v.at[0].set(coord[0] + jnp.where(is_node[0], nodes[0,5], 0.))
        v = v.at[1].set(coord[1] + jnp.where(is_node[1], nodes[1,5], 0.))
        v = v.at[2].set(coord[2] + jnp.where(is_node[2], nodes[2,5], 0.))
        v = v.at[3].set(coord[3] + jnp.where(is_node[3], nodes[3,5], 0.))
        
        def scan_fn(v_ci, ci):
            v, ci_val = v_ci
            has = is_conn[ci]
            si = jnp.argmax(is_node * (jnp.abs(nodes[:,0]-conns[ci,1])<0.01))
            di = jnp.argmax(is_node * (jnp.abs(nodes[:,0]-conns[ci,2])<0.01))
            update = conns[ci,3] * jnp.tanh(v[si]) + nodes[di,5]
            v = v.at[di].add(update * has)
            return (v, ci), None
        
        (v, _), _ = lax.scan(scan_fn, (v, 0), jnp.arange(100))
        last = jnp.clip(jnp.sum(is_node) - 1, 0).astype(jnp.int32)
        return jnp.tanh(v[last])
    
    return vmap(eval_one)(coords)

def make_substrate(n_obs=27, n_hid=10, n_act=8):
    n_obs += 1
    ih = jnp.zeros((n_obs * n_hid, 4))
    for i in range(n_obs):
        for h in range(n_hid):
            idx = i * n_hid + h
            ih = ih.at[idx].set(jnp.array([i/n_obs*2-1, -1., h/n_hid*2-1, 0.]))
    ho = jnp.zeros((n_hid * n_act, 4))
    for h in range(n_hid):
        for o in range(n_act):
            idx = h * n_act + o
            ho = ho.at[idx].set(jnp.array([h/n_hid*2-1, 0., o/n_act*2-1, 1.]))
    return ih, ho

SUBS_IH, SUBS_HO = make_substrate()

@jit
def genome_to_policy(nodes, conns):
    w_ih = cppn_query(nodes, conns, SUBS_IH).reshape(28, 10)
    w_ho = cppn_query(nodes, conns, SUBS_HO).reshape(10, 8)
    return {'w_ih': w_ih, 'w_ho': w_ho}

def policy_forward(params, obs):
    h = jnp.tanh(obs @ params['w_ih'][:-1] + params['w_ih'][-1])
    return jnp.tanh(h @ params['w_ho'])
