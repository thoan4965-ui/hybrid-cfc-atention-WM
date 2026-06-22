"""V2.6 full pipeline — NO slices in .at. Only scalar index or jnp.where."""
import jax, jax.numpy as jnp, time, numpy as np
from jax import random, jit, vmap, lax
from v2_6.genome import init_pop, mutate, MAX_GENES
from v2_6.cppn import genome_to_policy, policy_forward
from v2_6.env_ant import NoRewardAnt

env = NoRewardAnt(backend='mjx', energy_init=100., energy_cost=0.2, torque_cost=0.05)

@jit
def eval_batch(nodes, conns, keys):
    def single(n, c, k):
        pol = genome_to_policy(n, c)
        def step(s, _):
            s2 = env.step(s, policy_forward(pol, s.obs))
            return s2, s2.done
        _, dones = lax.scan(step, env.reset(k), jnp.arange(500))
        fd = jnp.argmax(dones > 0.5)
        return jnp.where(jnp.any(dones > 0.5), fd + 1, 500.)
    return vmap(single)(nodes, conns, keys)

@jit
def select_crossover(nodes, conns, fitness, key):
    pop = nodes.shape[0]
    k = random.split(key, 3)
    idx = random.randint(k[0], (pop, 3), 0, pop)
    best = jnp.argmax(fitness[idx], axis=1)
    parents = idx[jnp.arange(pop), best]
    mn = random.bernoulli(k[1], 0.5, (pop, MAX_GENES, 1))
    mc = random.bernoulli(k[2], 0.5, (pop, MAX_GENES, 1))
    return jnp.where(mn, nodes[parents], nodes), jnp.where(mc, conns[parents], conns)

def render_agent(nodes, conns, n_frames=200):
    pol = genome_to_policy(nodes, conns)
    state = env.reset(random.PRNGKey(0))
    frames = []
    for _ in range(n_frames):
        r = env.render(state)
        if r is not None and hasattr(r, 'shape') and r.size > 1:
            frames.append(np.array(r))
        state = env.step(state, policy_forward(pol, state.obs))
    return jnp.stack(frames) if len(frames) > 1 else None

def run(n_gen=200, pop_size=128, seed=3072):
    key = random.PRNGKey(seed)
    state = init_pop(key, pop_size)
    curve = []; t0 = time.time()
    _ = eval_batch(state['nodes'][:4], state['conns'][:4],
                   vmap(lambda i: random.PRNGKey(i))(jnp.arange(4)))
    print(f"Ready: {n_gen} gens x {pop_size} pop")
    
    for gen in range(n_gen):
        k = random.split(random.PRNGKey(gen), pop_size)
        fitness = eval_batch(state['nodes'], state['conns'], k)
        curve.append((float(jnp.max(fitness)), float(jnp.mean(fitness))))
        
        k1, k2 = random.split(random.PRNGKey(gen + seed))
        cn, cc = select_crossover(state['nodes'], state['conns'], fitness, k1)
        cn, cc, ni = mutate(cn, cc, k2, state['innov'])
        
        # Elitism — .at[int].set() (scalar index, safe)
        top2 = jnp.argsort(fitness)[-2:]
        cn = cn.at[0].set(state['nodes'][top2[0]])
        cn = cn.at[1].set(state['nodes'][top2[1]])
        cc = cc.at[0].set(state['conns'][top2[0]])
        cc = cc.at[1].set(state['conns'][top2[1]])
        
        state['nodes'] = cn; state['conns'] = cc; state['innov'] = ni
        
        if (gen + 1) % 20 == 0:
            dt = time.time() - t0; eta = dt/(gen+1)*(n_gen-gen-1)/60
            print(f"G{gen+1}: max={curve[-1][0]:.0f} mean={curve[-1][1]:.0f} [{dt:.0f}s ETA {eta:.0f}m]")
    
    k = random.split(random.PRNGKey(999), pop_size)
    ff = eval_batch(state['nodes'], state['conns'], k)
    best = int(jnp.argmax(ff))
    print(f"Done! Best: {float(jnp.max(ff)):.0f}")
    return {'curve': jnp.array(curve), 'best_nodes': state['nodes'][best],
            'best_conns': state['conns'][best], 'best_fitness': float(jnp.max(ff))}
