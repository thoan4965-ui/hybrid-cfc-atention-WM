"""V2.6 genome — ZERO .at on viewed arrays. Pure jnp.where + broadcasting."""
import jax, jax.numpy as jnp
from jax import random

MAX_GENES = 100; NODE_PARAMS = 7; CONN_PARAMS = 8
COL_INNOV=0; COL_TYPE=1; COL_X=2; COL_Y=3; COL_ACT=4; COL_BIAS=5; COL_EXPR=6

def init_pop(key, pop_size=128):
    nodes = jnp.full((pop_size, MAX_GENES, NODE_PARAMS), jnp.nan)
    conns = jnp.full((pop_size, MAX_GENES, CONN_PARAMS), jnp.nan)
    innov = 0
    for i in range(pop_size):
        for j in range(3):
            nodes = nodes.at[i, j].set(jnp.array([float(innov), 0, -1.0+j*1.0, -1.0+j*1.0, 0., 1., 1.]))
            innov += 1
        for j in range(2):
            conns = conns.at[i, j].set(jnp.array([float(innov), float(j), float(j+1), 0.5, 1., 1., 0., 0.]))
            innov += 1
    return {'nodes': nodes, 'conns': conns, 'innov': innov, 'pop_size': pop_size}

def mutate(nodes, conns, key, innov_start, subst=0.1, ins=0.03, dele=0.02):
    pop_size = nodes.shape[0]; ks = random.split(key, pop_size)
    innov = int(innov_start)
    
    for i in range(pop_size):
        k = ks[i]
        na = int(jnp.sum(~jnp.isnan(nodes[i, :, 0])))
        ca = int(jnp.sum(~jnp.isnan(conns[i, :, 0])))
        k1, k2, k3, k4 = random.split(k, 4)
        
        # COPY preserving NaN
        n_new = jnp.where(jnp.isnan(nodes[i]), jnp.nan, nodes[i])
        c_new = jnp.where(jnp.isnan(conns[i]), jnp.nan, conns[i])
        
        # --- 1. SUBSTITUTION ---
        sm = random.uniform(k1, (MAX_GENES,)) < subst
        noise_b = random.normal(k2, (MAX_GENES,)) * 0.05
        noise_w = random.normal(k3, (MAX_GENES,)) * 0.1
        
        col5 = jnp.arange(NODE_PARAMS) == COL_BIAS
        col3 = jnp.arange(CONN_PARAMS) == 3
        n_delta = (sm * noise_b)[:, None] * col5[None, :]
        c_delta = (sm * noise_w)[:, None] * col3[None, :]
        
        n_new = n_new + n_delta
        c_new = c_new + c_delta
        
        # --- 2. INSERTION (gene duplication) ---
        do_ins = float(random.uniform(k4, ())) < ins
        if do_ins and na < MAX_GENES - 1 and ca < MAX_GENES - 1:
            src = int(random.randint(k1, (), 0, max(na, 1)))
            src_row = n_new[src]
            mask_i = jnp.arange(NODE_PARAMS) == COL_INNOV
            mask_b = jnp.arange(NODE_PARAMS) == COL_BIAS
            new_row = jnp.where(mask_i, float(innov), src_row)
            new_row = jnp.where(mask_b, src_row[COL_BIAS] + float(random.normal(k2, ()) * 0.1), new_row)
            n_new = jnp.where((jnp.arange(MAX_GENES) == na)[:, None], new_row[None, :], n_new)
            innov += 1
            
            sc = int(random.randint(k3, (), 0, max(ca, 1)))
            src_c = c_new[sc]
            mask_ci = jnp.arange(CONN_PARAMS) == COL_INNOV
            mask_cw = jnp.arange(CONN_PARAMS) == 3
            new_conn = jnp.where(mask_ci, float(innov), src_c)
            new_conn = jnp.where(mask_cw, src_c[3] + float(random.normal(k4, ()) * 0.1), new_conn)
            c_new = jnp.where((jnp.arange(MAX_GENES) == ca)[:, None], new_conn[None, :], c_new)
            innov += 1
        
        # --- 3. DELETION ---
        if ca > 1:
            dm = random.uniform(k1, (MAX_GENES,)) < dele
            c_new = jnp.where(dm[:, None], jnp.nan, c_new)
        
        nodes = nodes.at[i].set(n_new)
        conns = conns.at[i].set(c_new)
    
    return nodes, conns, innov

print("✅ genome.py v7 — NaN-preserving copy")
