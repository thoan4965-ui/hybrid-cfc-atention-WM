"""Genome + JIT mutate + vmap crossover + Tag + Dopamine(5) + Module + Non-coding + Dup."""
import jax, jax.numpy as jnp
from jax import random, jit, vmap, lax, nn

MAX_GENES = 100; NODE_PARAMS = 8; CONN_PARAMS = 8; TAG_DIM = 16

def init_pop(key, pop_size=128):
    nodes = jnp.full((pop_size, MAX_GENES, NODE_PARAMS), jnp.nan)
    conns = jnp.full((pop_size, MAX_GENES, CONN_PARAMS), jnp.nan)
    tags = jnp.zeros((pop_size, TAG_DIM))
    innov = 0
    for i in range(pop_size):
        for j in range(3):
            nodes = nodes.at[i, j].set(jnp.array([float(innov), 0, -1.+j, -1.+j, 0., 1., 1., float(j)]))
            innov += 1
        for j in range(2):
            conns = conns.at[i, j].set(jnp.array([float(innov), float(j), float(j+1), 0.5, 1., 1., float(j), 0.]))
            innov += 1
    return {'nodes': nodes, 'conns': conns, 'tags': tags, 'innov': innov, 'pop_size': pop_size}

@jit
def mutate(nodes, conns, key, innov_start, subst=0.1, ins=0.03, dele=0.02):
    ps = nodes.shape[0]
    keys = vmap(lambda i: random.fold_in(key, i))(jnp.arange(ps, dtype=jnp.int32))
    def step(carry, x):
        n, c, innov = carry
        i, k = x
        na = jnp.sum(~jnp.isnan(n[i, :, 0]))
        ca = jnp.sum(~jnp.isnan(c[i, :, 0]))
        k1, k2, k3, k4 = random.split(k, 4)
        nn = jnp.where(jnp.isnan(n[i]), jnp.nan, n[i])
        cc = jnp.where(jnp.isnan(c[i]), jnp.nan, c[i])
        sm = random.uniform(k1, (MAX_GENES,)) < subst
        nn += (sm * random.normal(k2, (MAX_GENES,)) * 0.05)[:, None] * (jnp.arange(NODE_PARAMS) == 5)[None, :]
        cc += (sm * random.normal(k3, (MAX_GENES,)) * 0.1)[:, None] * (jnp.arange(CONN_PARAMS) == 3)[None, :]
        # Non-coding: mutate expr (column 5 for nodes, column 5 for conns)
        sm_expr = random.uniform(k1, (MAX_GENES,)) < 0.02
        nn += (sm_expr * random.normal(k2, (MAX_GENES,)) * 0.1)[:, None] * (jnp.arange(NODE_PARAMS) == 5)[None, :]
        cc += (sm_expr * random.normal(k3, (MAX_GENES,)) * 0.1)[:, None] * (jnp.arange(CONN_PARAMS) == 5)[None, :]
        # Module_id mutation (column 7 for nodes)
        sm_mod = random.uniform(k1, (MAX_GENES,)) < 0.01
        nn += (sm_mod * random.randint(k2, (MAX_GENES,), 0, 8))[:, None] * (jnp.arange(NODE_PARAMS) == 7)[None, :]
        # Insertion
        do_ins = (random.uniform(k4, ()) < ins) & (na < MAX_GENES - 1) & (ca < MAX_GENES - 1)
        src = random.randint(k1, (), 0, jnp.maximum(na, 1).astype(jnp.int32))
        nr = jnp.where(jnp.arange(NODE_PARAMS) == 0, innov.astype(jnp.float32), nn[src])
        nr = jnp.where(jnp.arange(NODE_PARAMS) == 5, nr[5] + random.normal(k2, ()) * 0.1, nr)
        nr = jnp.where(jnp.arange(NODE_PARAMS) == 7, nn[src, 7], nr)
        ins_pos = na
        nn = jnp.where(do_ins & (jnp.arange(MAX_GENES) == ins_pos)[:, None], nr[None, :], nn)
        sc = random.randint(k3, (), 0, jnp.maximum(ca, 1).astype(jnp.int32))
        nc = jnp.where(jnp.arange(CONN_PARAMS) == 0, (innov + 1).astype(jnp.float32), cc[sc])
        nc = jnp.where(jnp.arange(CONN_PARAMS) == 3, nc[3] + random.normal(k4, ()) * 0.1, nc)
        nc = nc.at[6].set(nn[src, 7])
        cc = jnp.where(do_ins & (jnp.arange(MAX_GENES) == ca)[:, None], nc[None, :], cc)
        added = do_ins * 2
        ca_fix = jnp.maximum(ca - 1, 0)
        cc = jnp.where((random.uniform(k1, (MAX_GENES,)) < dele)[:, None] & (ca_fix > 0), jnp.nan, cc)
        # Gene duplication: copy node at src + its connections
        do_dup = (random.uniform(k1, ()) < 0.02) & (na < MAX_GENES - 2) & (ca < MAX_GENES - 2)
        dup_src = random.randint(k1, (), 0, jnp.maximum(na, 1).astype(jnp.int32))
        dup_pos = na + do_ins.astype(jnp.int32)
        dup_nr = jnp.where(jnp.arange(NODE_PARAMS) == 0, (innov + added).astype(jnp.float32), nn[dup_src])
        nn = jnp.where(do_dup & (jnp.arange(MAX_GENES) == dup_pos)[:, None], dup_nr[None, :], nn)
        # Find connections from dup_src and copy them
        dup_conns = jnp.where((jnp.abs(c[i, :, 1] - (dup_src).astype(jnp.float32)) < 0.5)[:, None], c[i], jnp.nan)
        dup_conns_off = jnp.where(jnp.isnan(dup_conns[:, 0])[:, None], jnp.nan, dup_conns)
        dup_conns_off = jnp.where(jnp.arange(CONN_PARAMS)[None, :] == 0, (innov + added).astype(jnp.float32), dup_conns_off)
        dup_conns_off = jnp.where(jnp.arange(CONN_PARAMS)[None, :] == 1, (na + do_ins.astype(jnp.int32)).astype(jnp.float32), dup_conns_off)
        cc = jnp.where(do_dup & (jnp.arange(MAX_GENES) == (ca + do_ins.astype(jnp.int32)))[:, None], dup_conns_off, cc.astype(jnp.float32))
        added_dup = do_dup * 2
        n = n.at[i].set(nn); c = c.at[i].set(cc)
        return (n, c, innov + added.astype(jnp.int32) + added_dup.astype(jnp.int32)), None
    (nodes, conns, innov), _ = lax.scan(step, (nodes, conns, jnp.int32(innov_start)),
                                         (jnp.arange(ps, dtype=jnp.int32), keys))
    return nodes, conns, innov

def mutate_tags(t, key, noise=0.03, rate=0.05):
    k1, k2 = random.split(key); mask = random.uniform(k1, t.shape) < rate
    return t + mask * random.normal(k2, t.shape) * noise

def crossover_tags(t1, t2, key):
    pick = random.bernoulli(key, shape=(TAG_DIM,))
    return jnp.where(pick, t1, t2)

@jit
def crossover_innov(child_n, child_c, p1n, p1c, p2n, p2c, fit1, fit2, key):
    ci = child_n[:, 0]; p1i = p1n[:, 0][None, :]; p2i = p2n[:, 0][None, :]
    m1 = jnp.abs(p1i - ci[:, None]) < 0.5; m2 = jnp.abs(p2i - ci[:, None]) < 0.5
    i1 = jnp.argmax(m1, 1); i2 = jnp.argmax(m2, 1)
    h1 = jnp.any(m1, 1); h2 = jnp.any(m2, 1)
    # Module-aware crossover: align by innov + module_id
    mod1 = p1n[i1, 7] if p1n.shape[1] > 7 else jnp.zeros(MAX_GENES)
    mod2 = p2n[i2, 7] if p2n.shape[1] > 7 else jnp.zeros(MAX_GENES)
    pk = random.bernoulli(key, shape=(MAX_GENES,))
    ma = h1 & h2 & (mod1 == mod2); o1 = h1 & ~h2; o2 = ~h1 & h2; sf = jnp.where(fit1 >= fit2, 1., 0.)
    cn = jnp.where(ma[:, None], jnp.where(pk[:, None], p1n[i1], p2n[i2]),
                   jnp.where(o1[:, None], jnp.where(sf, p1n[i1], jnp.nan),
                             jnp.where(o2[:, None], jnp.where(1-sf, p2n[i2], jnp.nan), child_n)))
    ci = child_c[:, 0]; p1i = p1c[:, 0][None, :]; p2i = p2c[:, 0][None, :]
    m1 = jnp.abs(p1i - ci[:, None]) < 0.5; m2 = jnp.abs(p2i - ci[:, None]) < 0.5
    i1 = jnp.argmax(m1, 1); i2 = jnp.argmax(m2, 1)
    h1 = jnp.any(m1, 1); h2 = jnp.any(m2, 1)
    ma = h1 & h2; o1 = h1 & ~h2; o2 = ~h1 & h2
    cc = jnp.where(ma[:, None], jnp.where(pk[:, None], p1c[i1], p2c[i2]),
                   jnp.where(o1[:, None], jnp.where(sf, p1c[i1], jnp.nan),
                             jnp.where(o2[:, None], jnp.where(1-sf, p2c[i2], jnp.nan), child_c)))
    return cn, cc

# === Dopamine — 2nd genome (5 floats per agent) ===
def init_dopas(key, pop_size):
    return random.normal(key, (pop_size, 5)) * 0.5

@jit
def mutate_dopas(dopas, key, noise=0.3):
    return dopas + noise * random.normal(key, dopas.shape)

@jit
def crossover_dopas(d1, d2, key):
    pick = random.bernoulli(key, shape=(5,))
    return jnp.where(pick, d1, d2)

# === Regulatory — 3rd genome (16 floats per agent: 8 module enable + 8 per-module mutation rate) ===
REG_DIM = 16
REG_MOD_DIM = 8

def init_regs(key, pop_size):
    return random.normal(key, (pop_size, REG_DIM)) * 0.5

@jit
def mutate_regs(regs, key, noise=0.3):
    return regs + noise * random.normal(key, regs.shape)

@jit
def crossover_regs(r1, r2, key):
    pick = random.bernoulli(key, shape=(REG_DIM,))
    return jnp.where(pick, r1, r2)

def regs_module_enable(regs):
    """First 8 floats: sigmoid > 0.5 → module on/off."""
    return jax.nn.sigmoid(regs[:, :REG_MOD_DIM]) > 0.5

def regs_module_rate(regs):
    """Next 8 floats: sigmoid → per-module mutation rate scale [0.1, 2.0]."""
    return 0.1 + 1.9 * jax.nn.sigmoid(regs[:, REG_MOD_DIM:])
