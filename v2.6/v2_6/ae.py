"""Autoencoder 10→16→10 — nan-safe, ko đổi dù thêm sensor/body."""
import jax, jax.numpy as jnp
from jax import random, jit, vmap

def init_ae(key):
    k1, k2 = random.split(key)
    return {'We': random.normal(k1, (10, 16)) * 0.01, 'be': jnp.zeros(16),
            'Wd': random.normal(k2, (16, 10)) * 0.01, 'bd': jnp.zeros(10)}

def encode(p, x): return jnp.tanh(x @ p['We'] + p['be'])
def decode(p, z): return z @ p['Wd'] + p['bd']

@jit
def train_ae(p, d, key, lr=0.001, steps=30):
    d = jnp.where(jnp.isnan(d), 0., d)
    dm = jnp.mean(d); ds = jnp.std(d) + 1e-3; dn = (d - dm) / ds; dn = jnp.clip(dn, -3., 3.)
    def loss(p):
        z = jnp.tanh(dn @ p['We'] + p['be']); recon = z @ p['Wd'] + p['bd']
        return jnp.mean((dn - recon) ** 2) + 1e-3 * (jnp.sum(p['We']**2) + jnp.sum(p['Wd']**2))
    def step(p, _):
        g = jax.grad(loss)(p); g = {k: jnp.clip(v, -1., 1.) for k, v in g.items()}
        return {k: p[k] - lr * g[k] for k in p}, None
    fp, _ = jax.lax.scan(step, p, None, length=steps)
    return fp
