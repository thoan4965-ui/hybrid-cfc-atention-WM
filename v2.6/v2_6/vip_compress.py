"""Compress teacher weights → genome via CPPN bottleneck."""
import jax, jax.numpy as jnp, numpy as np, os
jax.config.update('jax_default_matmul_precision', 'high')
from jax import random, jit, vmap, grad
from v2_6.cppn import genome_to_policy
from v2_6.genome import init_pop, MAX_GENES, NODE_PARAMS, CONN_PARAMS

def teacher_weights_to_target(params):
    """Flatten teacher weights into 1 target vector. Match CPPN output order."""
    return jnp.concatenate([
        params['w_ih'].ravel(), params['w_ho'].ravel(),
        params['w_pred'].ravel(),
        jnp.zeros(46*10 + 16*10)])  # teacher ko có w_spat/w_thought → zeros

def genome_to_weights(nodes, conns):
    """Run CPPN → policy weights → flatten."""
    pol = genome_to_policy(nodes, conns)
    return jnp.concatenate([
        pol['w_ih'].ravel(), pol['w_ho'].ravel(),
        pol['w_pred'].ravel(), pol['w_spat'].ravel(),
        pol['w_thought'].ravel()])

@jit
def compression_loss(nodes, conns, target):
    """MSE between CPPN output and teacher target weights."""
    pred = genome_to_weights(nodes, conns)
    return jnp.mean((pred - target) ** 2)

def compress_teacher(teacher_params, n_opt_steps=500, lr=0.01, seed=3072,
                     pop_size=1):
    """Optimize genome to minimize ||CPPN(genome) - teacher_weights||²."""
    target = teacher_weights_to_target(teacher_params)
    key = random.PRNGKey(seed)

    # Start from a random genome
    state = init_pop(key, pop_size)
    nodes = state['nodes'][0:1]
    conns = state['conns'][0:1]

    for step in range(n_opt_steps):
        g = grad(lambda n, c: compression_loss(n, c, target))(nodes, conns)
        nodes = nodes - lr * jnp.clip(g[0], -0.1, 0.1)
        conns = conns - lr * jnp.clip(g[1], -0.1, 0.1)
        loss = compression_loss(nodes, conns, target)

        if (step + 1) % 100 == 0:
            print(f"  Compress step{step+1}: loss={float(loss):.6f}")

    return {'nodes': nodes, 'conns': conns}

def save_vip_genome(genome, path="vip_genome.npz"):
    """Save VIP genome to file."""
    np.savez(path,
             nodes=np.array(genome['nodes']),
             conns=np.array(genome['conns']))
    print(f"  VIP genome saved to {path}")

def load_vip_genome(path="vip_genome.npz"):
    """Load VIP genome from file."""
    d = np.load(path)
    return {'nodes': jnp.array(d['nodes']), 'conns': jnp.array(d['conns'])}
