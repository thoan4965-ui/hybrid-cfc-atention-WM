"""V2.9 — GA + Gradient + Hebbian + Dopamine + HF checkpoint."""
import jax, jax.numpy as jnp, time, numpy as np, os
from jax import random, jit, vmap, lax
from v2_6.genome import (init_pop, mutate, crossover_innov, mutate_tags,
                          crossover_tags, init_dopas, mutate_dopas, crossover_dopas,
                          MAX_GENES, NODE_PARAMS, CONN_PARAMS, TAG_DIM)
from v2_6.cppn import genome_to_policy, policy_forward
from v2_6.env_ant import NoRewardAnt
from v2_6.ae import init_ae, train_ae, encode, decode
from v2_6.hebbian import hebbian_update

from huggingface_hub import HfApi

env = NoRewardAnt(backend='mjx', energy_init=20., energy_cost=0.4, torque_cost=0.05)

def pred_loss(w_ih, w_pred, obs, target):
    h = jnp.tanh(obs @ w_ih[:-1] + w_ih[-1])
    pred = h @ w_pred
    return jnp.mean((target - pred) ** 2)

@jit
def eval_batch(nodes, conns, dopas, keys):
    def single(n, c, d, k):
        pol = genome_to_policy(n, c)
        pol['w_dopa'] = d
        def step(ps, _):
            pol, s = ps
            a, pred_n = policy_forward(pol, s.obs)
            s2 = env.step(s, a)

            wg, wh, wa = jax.nn.softmax(pol['w_dopa'] * 3)
            w_grad, w_hebb, w_ga = wg, wh, wa

            pol = hebbian_update(pol, s.obs, scale=w_hebb)

            g_ih, g_pred = jax.grad(pred_loss, argnums=(0,1))(
                pol['w_ih'], pol['w_pred'], s.obs, s2.obs)
            lr = 0.001
            pol['w_ih'] = pol['w_ih'] - lr * w_grad * jnp.clip(g_ih, -1., 1.)
            pol['w_pred'] = pol['w_pred'] - lr * w_grad * jnp.clip(g_pred, -1., 1.)

            pol['w_dopa'] = jnp.array([w_grad, w_hebb, w_ga])

            return (pol, s2), (s2.done, jnp.concatenate([s.obs, a, s2.info['energy'][None]]))
        (pol, s_final), (d, ex) = lax.scan(step, (pol, env.reset(k)), jnp.arange(500))
        d = jnp.nan_to_num(d, nan=1.)
        ex = jnp.nan_to_num(ex, nan=0.)
        fd = jnp.argmax(d > 0.5)
        alive = jnp.where(jnp.any(d > 0.5), fd + 1., 500.)
        final_energy = jnp.nan_to_num(s_final.info['energy'], nan=0.)
        dopa = pol['w_dopa']
        return (alive, dopa), jnp.concatenate([alive[None], final_energy[None], jnp.mean(ex, 0)])
    (alive_arr, dopa_arr), r = vmap(single)(nodes, conns, dopas, keys)
    return alive_arr, dopa_arr, r

def save_checkpoint(state, ae, gen, curve, path, hf_api=None, hf_repo=None):
    np.savez(path,
        nodes=np.array(state['nodes']), conns=np.array(state['conns']),
        tags=np.array(state['tags']), dopas=np.array(state['dopas']), innov=state['innov'],
        ae_We=np.array(ae['We']), ae_be=np.array(ae['be']),
        ae_Wd=np.array(ae['Wd']), ae_bd=np.array(ae['bd']),
        gen=gen, curve=np.array(curve))
    if hf_api and hf_repo:
        try:
            hf_api.upload_file(path_or_fileobj=path, path_in_repo=f"checkpoints/v2.9/cp_{gen}.npz", repo_id=hf_repo)
            print(f"  Checkpoint G{gen} uploaded to HF", flush=True)
        except:
            print(f"  Warning: HF upload failed for G{gen}, checkpoint saved locally", flush=True)

def load_checkpoint(path):
    d = np.load(path, allow_pickle=True)
    state = {'nodes': jnp.array(d['nodes']), 'conns': jnp.array(d['conns']),
             'tags': jnp.array(d['tags']), 'innov': int(d['innov']),
             'pop_size': d['nodes'].shape[0]}
    if 'dopas' in d:
        state['dopas'] = jnp.array(d['dopas'])
    ae = {'We': jnp.array(d['ae_We']), 'be': jnp.array(d['ae_be']),
          'Wd': jnp.array(d['ae_Wd']), 'bd': jnp.array(d['ae_bd'])}
    return state, ae, int(d['gen']), list(d['curve'])

def download_latest_hf(api, repo_id, dest="."):
    try:
        files = api.list_repo_files(repo_id, token=api.token)
        cps = sorted([f for f in files if f.startswith("checkpoints/v2.9/cp_") and f.endswith(".npz")])
        if not cps: return None
        return api.hf_hub_download(repo_id=repo_id, filename=cps[-1], local_dir=dest, token=api.token)
    except:
        return None

def run(n_gen=200, pop_size=128, seed=3072, resume_path=None):
    key = random.PRNGKey(seed)
    hf_api = None
    try:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            hf_api = HfApi(token=hf_token); print(f"HF OK: {hf_api.whoami()['name']}", flush=True)
    except: pass

    if resume_path:
        state, ae, gen_start, curve = load_checkpoint(resume_path)
        if 'dopas' not in state:
            state['dopas'] = init_dopas(random.PRNGKey(seed), state['pop_size'])
        print(f"Resumed G{gen_start}, pop={state['pop_size']}", flush=True)
    else:
        ae_key, ga_key = random.split(key)
        state = init_pop(ga_key, pop_size)
        state['dopas'] = init_dopas(ga_key, pop_size)
        ae = init_ae(ae_key)
        gen_start = 0; curve = []
        n4 = vmap(lambda i: random.PRNGKey(i))(jnp.arange(4))
        f4, _, _ = eval_batch(state['nodes'][:4], state['conns'][:4], state['dopas'][:4], n4)
        print(f"V2.9+C+A: {n_gen}gen x {pop_size}pop", flush=True)

    t0 = time.time()
    k0 = random.PRNGKey(gen_start * 3)
    for g in range(gen_start, n_gen):
        k0 = random.PRNGKey(g * 3)
        fs, dopas, rs = [], [], []
        for ri in range(3):
            kk = random.split(random.fold_in(k0, ri), pop_size)
            f_arr, d_arr, r = eval_batch(state['nodes'], state['conns'], state['dopas'], kk)
            fs.append(f_arr); dopas.append(d_arr); rs.append(r)
        f = jnp.nan_to_num(jnp.mean(jnp.stack(fs), axis=0), nan=0.)
        dopa_mean = jnp.mean(jnp.stack(dopas), axis=(0,1))
        r = jnp.mean(jnp.stack(rs), axis=0)
        final_e = jnp.nan_to_num(r[:, 1], nan=0.) if r.ndim > 1 else jnp.zeros(pop_size)
        ex_raw = r[:, 2:] if r.ndim > 1 and r.shape[1] > 2 else jnp.zeros((pop_size, 37))

        ae_input = jnp.concatenate([
            ex_raw[:, 29:37], f[:, None] / 500., ex_raw[:, 37:38]], axis=1)

        skip_ae = float(jnp.mean(f)) < 7.
        if not skip_ae:
            ae = train_ae(ae, ae_input, random.PRNGKey(g + 1000))
            dm_ae = jnp.mean(ae_input); ds_ae = jnp.std(ae_input) + 1e-3
            ae_norm = jnp.clip((ae_input - dm_ae) / ds_ae, -3., 3.)
            nt = vmap(lambda e: encode(ae, e))(ae_norm)
            all_dec = vmap(lambda e: decode(ae, encode(ae, e)))(ae_norm)
            per_loss = jnp.mean((ae_norm - all_dec) ** 2, axis=1)
            range_ok = (jnp.max(per_loss) - jnp.min(per_loss)) > 0.05
            dopamine = jnp.where(range_ok, (per_loss - jnp.min(per_loss)) / (jnp.max(per_loss) - jnp.min(per_loss) + 1e-8), 0.)
            ae_loss = jnp.mean(per_loss)
        else:
            nt = jnp.zeros((pop_size, TAG_DIM)); dopamine = jnp.zeros(pop_size); ae_loss = 0.
        f_total = f + 5 * dopamine
        curve.append((float(jnp.max(f_total)), float(jnp.mean(f_total))))

        w_ga = float(dopa_mean[2])
        mr = w_ga
        k1, k2 = random.split(random.PRNGKey(g + seed))
        idx = random.randint(k1, (pop_size, 3), 0, pop_size)
        pr = idx[jnp.arange(pop_size), jnp.argmax(f_total[idx], axis=1)]
        np2 = pop_size // 2
        p1_arr = pr[0::2]; p2_arr = pr[1::2]
        kx_arr = vmap(lambda i: random.fold_in(k2, i * 4 + 0))(jnp.arange(np2))
        ky_arr = vmap(lambda i: random.fold_in(k2, i * 4 + 1))(jnp.arange(np2))
        kt_arr = vmap(lambda i: random.fold_in(k2, i * 4 + 2))(jnp.arange(np2))
        ks_arr = vmap(lambda i: random.fold_in(k2, i * 4 + 3))(jnp.arange(np2))
        cx_batch = vmap(crossover_innov, in_axes=(0,0,0,0,0,0,0,0,0))
        c1n, c1c = cx_batch(state['nodes'][p1_arr], state['conns'][p1_arr],
            state['nodes'][p1_arr], state['conns'][p1_arr],
            state['nodes'][p2_arr], state['conns'][p2_arr],
            f_total[p1_arr], f_total[p2_arr], kx_arr)
        c2n, c2c = cx_batch(state['nodes'][p2_arr], state['conns'][p2_arr],
            state['nodes'][p1_arr], state['conns'][p1_arr],
            state['nodes'][p2_arr], state['conns'][p2_arr],
            f_total[p1_arr], f_total[p2_arr], ky_arr)
        cn = jnp.full((pop_size, MAX_GENES, NODE_PARAMS), jnp.nan)
        cc = jnp.full((pop_size, MAX_GENES, CONN_PARAMS), jnp.nan)
        tc = jnp.zeros((pop_size, TAG_DIM))
        cn = cn.at[0::2].set(c1n); cn = cn.at[1::2].set(c2n)
        cc = cc.at[0::2].set(c1c); cc = cc.at[1::2].set(c2c)
        ct_batch = vmap(crossover_tags)
        tc = tc.at[0::2].set(ct_batch(nt[p1_arr], nt[p2_arr], kt_arr))
        tc = tc.at[1::2].set(ct_batch(nt[p2_arr], nt[p1_arr], ks_arr))
        cd_batch = vmap(crossover_dopas)
        dc = jnp.zeros((pop_size, 3))
        dc = dc.at[0::2].set(cd_batch(state['dopas'][p1_arr], state['dopas'][p2_arr], kt_arr))
        dc = dc.at[1::2].set(cd_batch(state['dopas'][p2_arr], state['dopas'][p1_arr], ks_arr))
        td_arr = vmap(lambda p1, p2: jnp.tile(nt[p1] - nt[p2], 7)[:MAX_GENES])(p1_arr, p2_arr)
        delta = jnp.zeros((np2, MAX_GENES, NODE_PARAMS)).at[:, :, 5].set(0.01 * td_arr)
        cn = cn.at[0::2].set(cn[0::2] + delta)
        cn = cn.at[1::2].set(cn[1::2] + delta)
        cn, cc, ni_jit = mutate(cn, cc, k2, state['innov'], .1*mr, .03*mr, .02*mr)
        ni = int(ni_jit)
        tc = mutate_tags(tc, k2, noise=.03*mr)
        dc = mutate_dopas(dc, k2, noise=.03*mr)
        top2 = jnp.argsort(f_total)[-2:]
        for j in range(2):
            cn = cn.at[j].set(state['nodes'][top2[j]]); cc = cc.at[j].set(state['conns'][top2[j]])
            tc = tc.at[j].set(state['tags'][top2[j]])
            dc = dc.at[j].set(state['dopas'][top2[j]])
        state['nodes'] = cn; state['conns'] = cc; state['tags'] = tc; state['dopas'] = dc; state['innov'] = ni

        if (g + 1) % 20 == 0:
            dt = time.time() - t0; eta = dt/(g-gen_start+1)*(n_gen-g-1)/60
            print(f"G{g+1}: max={curve[-1][0]:.0f} mean={curve[-1][1]:.0f}"
                  f" rmx={float(jnp.max(f)):.0f} rmn={float(jnp.mean(f)):.0f}"
                  f" ae={ae_loss:.4f} dopa={dopa_mean[0]:.2f}/{dopa_mean[1]:.2f}/{dopa_mean[2]:.2f}"
                  f" [{dt:.0f}s ETA {eta:.0f}m]", flush=True)

        if (g + 1) % 500 == 0 or g == n_gen - 1:
            os.makedirs("checkpoints/v2.9", exist_ok=True)
            cp_path = f"checkpoints/v2.9/cp_{g+1}.npz"
            save_checkpoint(state, ae, g+1, curve, cp_path, hf_api, "hhian/checkpoints")

    ffs = []
    for ri in range(3):
        kk = random.split(random.fold_in(k0, ri), pop_size)
        ff, _, _ = eval_batch(state['nodes'], state['conns'], state['dopas'], kk)
        ffs.append(ff)
    ff = jnp.nan_to_num(jnp.mean(jnp.stack(ffs), axis=0), nan=0.)
    bi = int(jnp.argmax(ff))
    print(f"Best steps (raw): {float(jnp.max(ff)):.0f}", flush=True)
    os.makedirs("v2_6/results", exist_ok=True)
    ts = time.strftime("%m%d_%H%M")
    np.savez(f'v2_6/results/v29_{ts}.npz',
        curve=np.array(curve), best_nodes=np.array(state['nodes'][bi]),
        best_conns=np.array(state['conns'][bi]), best_fitness=float(jnp.max(ff)),
        best_total_fitness=float(curve[-1][0]))
    return {'curve': jnp.array(curve), 'best_nodes': state['nodes'][bi],
            'best_conns': state['conns'][bi], 'best_fitness': float(jnp.max(ff))}
