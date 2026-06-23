"""V2.6 Phase 3e — Fixed: energy=20, AE input = body+action+fitness+energy, dopamine bonus."""
import jax, jax.numpy as jnp, time, numpy as np
from jax import random, jit, vmap, lax
from v2_6.genome import (init_pop, mutate, crossover_innov, mutate_tags,
                          crossover_tags, MAX_GENES, NODE_PARAMS, CONN_PARAMS, TAG_DIM)
from v2_6.cppn import genome_to_policy, policy_forward
from v2_6.env_ant import NoRewardAnt
from v2_6.ae import init_ae, train_ae, encode, decode
from v2_6.hebbian import hebbian_update
from v2_6.expression import mechanism_x

env = NoRewardAnt(backend='mjx', energy_init=20., energy_cost=0.4, torque_cost=0.05)

@jit
def eval_batch(nodes, conns, keys):
    def single(n, c, k):
        pol = genome_to_policy(n, c)
        def step(ps, _):
            pol, s = ps
            a = policy_forward(pol, s.obs); s2 = env.step(s, a)
            pol = hebbian_update(pol, s.obs)
            return (pol, s2), (s2.done, jnp.concatenate([s.obs, a, s2.info['energy'][None]]))
        (pol, s_final), (d, ex) = lax.scan(step, (pol, env.reset(k)), jnp.arange(500))
        fd = jnp.argmax(d > 0.5)
        alive = jnp.where(jnp.any(d > 0.5), fd + 1., 500.)
        final_energy = s_final.info['energy']
        return alive, jnp.concatenate([alive[None], final_energy[None], jnp.mean(ex, 0)])
    return vmap(single)(nodes, conns, keys)

def run(n_gen=200, pop_size=128, seed=3072):
    key = random.PRNGKey(seed); ae_key, ga_key = random.split(key)
    state = init_pop(ga_key, pop_size); ae = init_ae(ae_key)
    curve = []; t0 = time.time()
    _ = eval_batch(state['nodes'][:4], state['conns'][:4],
                   vmap(lambda i: random.PRNGKey(i))(jnp.arange(4)))
    print(f"Phase 3e: {n_gen} gen x {pop_size} pop (energy=20, dopamine bonus)", flush=True)

    for g in range(n_gen):
        k0 = random.PRNGKey(g * 3)
        fs, es, exs = [], [], []
        for ri in range(3):
            kk = random.split(random.fold_in(k0, ri), pop_size)
            f, r = eval_batch(state['nodes'], state['conns'], kk)
            es.append(r[:, 1]); exs.append(r[:, 2:]); fs.append(f)
        f = jnp.mean(jnp.stack(fs), axis=0)
        final_e = jnp.mean(jnp.stack(es), axis=0)
        ex_raw = jnp.mean(jnp.stack(exs), axis=0)

        ae_input = jnp.concatenate([
            ex_raw[:, 29:37],       # action (8)
            f[:, None] / 500.,      # fitness [0,1]
            ex_raw[:, 37:38]        # energy [can be up to 20]
        ], axis=1)                   # total: 10 dim

        skip_ae = float(jnp.mean(f)) < 7.
        if not skip_ae:
            ae = train_ae(ae, ae_input, random.PRNGKey(g + 1000))
            nt = vmap(lambda e: encode(ae, e))(ae_input)
            all_dec = vmap(lambda e: decode(ae, encode(ae, e)))(ae_input)
            per_loss = jnp.mean((ae_input - all_dec) ** 2, axis=1)
            dopamine = per_loss / (jnp.max(per_loss) + 1e-8)
            ae_loss = jnp.mean(per_loss)
        else:
            nt = jnp.zeros((pop_size, TAG_DIM))
            dopamine = jnp.zeros(pop_size)
            ae_loss = 0.
        f_total = f + 50 * dopamine

        curve.append((float(jnp.max(f_total)), float(jnp.mean(f_total))))

        mx = mechanism_x(state['nodes'], state['tags'], f_total)
        mr = float(jnp.mean(mx))

        k1, k2 = random.split(random.PRNGKey(g + seed))
        idx = random.randint(k1, (pop_size, 3), 0, pop_size)
        pr = idx[jnp.arange(pop_size), jnp.argmax(f_total[idx], axis=1)]
        cn = jnp.full((pop_size, MAX_GENES, NODE_PARAMS), jnp.nan)
        cc = jnp.full((pop_size, MAX_GENES, CONN_PARAMS), jnp.nan)
        tc = jnp.zeros((pop_size, TAG_DIM))

        for i in range(pop_size // 2):
            p1, p2 = pr[i*2], pr[i*2+1]
            kx, ky, kt1, kt2 = random.split(random.fold_in(k2, i), 4)
            c1n, c1c = crossover_innov(state['nodes'][p1], state['conns'][p1],
                                        state['nodes'][p1], state['conns'][p1],
                                        state['nodes'][p2], state['conns'][p2],
                                        f_total[p1], f_total[p2], kx)
            c2n, c2c = crossover_innov(state['nodes'][p2], state['conns'][p2],
                                        state['nodes'][p1], state['conns'][p1],
                                        state['nodes'][p2], state['conns'][p2],
                                        f_total[p1], f_total[p2], ky)
            cn = cn.at[i*2].set(c1n); cn = cn.at[i*2+1].set(c2n)
            cc = cc.at[i*2].set(c1c); cc = cc.at[i*2+1].set(c2c)
            tc = tc.at[i*2].set(crossover_tags(nt[p1], nt[p2], kt1))
            tc = tc.at[i*2+1].set(crossover_tags(nt[p1], nt[p2], kt2))
            td = jnp.tile(nt[p1] - nt[p2], 7)[:MAX_GENES]
            delta = jnp.zeros((MAX_GENES, NODE_PARAMS)).at[:, 5].set(0.01 * td)
            cn = cn.at[i*2].set(cn[i*2] + delta)
            cn = cn.at[i*2+1].set(cn[i*2+1] + delta)

        cn, cc, ni = mutate(cn, cc, k2, state['innov'], .1*mr, .03*mr, .02*mr)
        tc = mutate_tags(tc, k2, noise=.03*mr)
        top2 = jnp.argsort(f_total)[-2:]
        for j in range(2):
            cn = cn.at[j].set(state['nodes'][top2[j]])
            cc = cc.at[j].set(state['conns'][top2[j]])
            tc = tc.at[j].set(state['tags'][top2[j]])
        state['nodes'] = cn; state['conns'] = cc
        state['tags'] = tc; state['innov'] = ni

        if (g + 1) % 20 == 0:
            dt = time.time() - t0; eta = dt/(g+1)*(n_gen-g-1)/60
            al = float(ae_loss)
            print(f"G{g+1}: max={curve[-1][0]:.0f} mean={curve[-1][1]:.0f}"
                  f" ae={al:.4f} mx={mr:.3f} [{dt:.0f}s ETA {eta:.0f}m]", flush=True)

    k0 = random.PRNGKey(999)
    ffs = []
    for ri in range(3):
        kk = random.split(random.fold_in(k0, ri), pop_size)
        f, _ = eval_batch(state['nodes'], state['conns'], kk)
        ffs.append(f)
    ff = jnp.mean(jnp.stack(ffs), axis=0)
    bi = int(jnp.argmax(ff))
    print(f"Best steps: {float(jnp.max(ff)):.0f}", flush=True)
    return {'curve': jnp.array(curve), 'best_nodes': state['nodes'][bi],
            'best_conns': state['conns'][bi], 'best_fitness': float(jnp.max(ff))}
