"""V2.9.x — GA + Gradient + Hebbian + Dopamine + Regulatory + Spatial + Planning + Diagnosis + Imitation.
   Feature flags: --spatial, --planning, --diagnosis, --imitation (mode=vip or mode=base)."""
import jax, jax.numpy as jnp, time, numpy as np, os
jax.config.update('jax_default_matmul_precision', 'high')
from jax import random, jit, vmap, lax
from v2_6.genome import (init_pop, mutate, crossover_innov, mutate_tags,
    crossover_tags, init_dopas, mutate_dopas, crossover_dopas,
    init_regs, mutate_regs, crossover_regs,
    init_spatial, mutate_spatial, crossover_spatial,
    init_plan, mutate_plan, crossover_plan,
    init_diag, mutate_diag, crossover_diag,
    init_mirror, mutate_mirror, crossover_mirror,
    init_thought, mutate_thought, crossover_thought,
    MAX_GENES, NODE_PARAMS, CONN_PARAMS, TAG_DIM,
    REG_DIM, SPATIAL_DIM, PLAN_DIM, DIAG_DIM, MIRROR_DIM, THOUGHT_DIM)
from v2_6.cppn import genome_to_policy, policy_forward, spatial_encoding
from v2_6.env_ant import NoRewardAnt
from v2_6.ae import init_ae, train_ae, encode, decode
from v2_6.hebbian import hebbian_update
from v2_6.train_teacher import train_teacher
from v2_6.vip_compress import compress_teacher, save_vip_genome, load_vip_genome
from huggingface_hub import HfApi

env = NoRewardAnt(backend='mjx', energy_init=20., energy_cost=0.4, torque_cost=0.05)
if hasattr(env, 'sys') and hasattr(env.sys, 'mj_model'):
    env.sys.mj_model.opt.iterations = 3
    env.sys.mj_model.opt.ls_iterations = 5
    env.sys.mj_model.opt.timestep = 0.001

def pred_loss(w_ih, w_pred, obs, target):
    h = jnp.tanh(obs @ w_ih[:-1] + w_ih[-1])
    pred = h @ w_pred
    return jnp.mean((target - pred) ** 2)

def make_eval_batch(flag_spatial=False, flag_planning=False, flag_diag=False, flag_mirror=False, flag_thought=False):
    """Factory: flags are Python constants (closure), not JIT-traced args."""
    TOKEN_DIM = 16
    B_MAX = 10
    L_MAX = 20

    @jit
    def eval_batch(nodes, conns, dopas, regs, spacials, plans, diags, mirrors, thoughts, keys, elite_data=None):
        def single(n, c, d, r, sp, pl, dg, mi, th, k):
            pol = genome_to_policy(n, c, regs=r)
            pol['w_dopa'] = d
            d0 = d

            # Planning: fixed shapes B_MAX×L_MAX, mask unused
            plan_noise = 0.1 + 0.4 * jax.nn.sigmoid(pl[2]) if flag_planning else 0.

            # Mirror params
            mirror_lr = 0.001 + 0.009 * jax.nn.sigmoid(mi[3]) if flag_mirror else 0.
            mirror_select = 1.0 + jax.nn.sigmoid(mi[2]) if flag_mirror else 99.

            # Thought params
            thought_scale = jax.nn.sigmoid(th[0]) * 0.5 if flag_thought else 0.
            thought_decay = 0.1 + 0.9 * jax.nn.sigmoid(th[1]) if flag_thought else 1.

            # Elite data: tuple (wih, fitness), JIT-safe
            use_elite = (elite_data is not None and flag_mirror)
            e_wih = elite_data[0] if use_elite else jnp.zeros(1)
            e_fit = elite_data[1] if use_elite else 0.

            def step(carry, _):
                pol, s, thought_state = carry
                obs_i = s.obs  # always 29-dim from env

                # Spatial encoding (no concat — passed to policy_forward)
                if flag_spatial:
                    x = s.pipeline_state.x.pos[0, 0]
                    y = s.pipeline_state.x.pos[0, 1]
                    spat_enc = spatial_encoding(x, y, scale=jax.nn.sigmoid(sp[0]))
                else:
                    spat_enc = None

                # Inner speech
                if flag_thought:
                    h_t = jnp.tanh(obs_i @ pol['w_ih'][:-1] + pol['w_ih'][-1])
                    new_thought = jnp.tanh(h_t[:, None] * jnp.ones(TOKEN_DIM)[None, :]).mean(0)
                    thought_state = thought_state * thought_decay + new_thought * (1 - thought_decay)
                    ts = thought_state * thought_scale
                else:
                    ts = None

                a, pred_n = policy_forward(pol, obs_i, spat_enc=spat_enc, thought_state=ts)
                s2 = env.step(s, a)
                s2 = s2.replace(obs=jnp.nan_to_num(s2.obs, 0.))

                pred_error = jnp.mean((s2.obs - pred_n) ** 2)
                adapt = jnp.tanh(jnp.abs(d0[3]) * pred_error)
                wg, wh, wa = jax.nn.softmax(d0[:3] + jnp.array([adapt, -adapt, 0.]))
                w_grad, w_hebb, w_ga = wg, wh, wa
                lr_grad = 0.001 * (1 + jnp.abs(d0[4]) * 9)

                pol = hebbian_update(pol, s.obs, scale=w_hebb)

                g_ih, g_pred = jax.grad(pred_loss, argnums=(0,1))(
                    pol['w_ih'], pol['w_pred'], obs_i, s2.obs)
                pol['w_ih'] = pol['w_ih'] - lr_grad * w_grad * jnp.clip(g_ih, -1., 1.)
                pol['w_pred'] = pol['w_pred'] - lr_grad * w_grad * jnp.clip(g_pred, -1., 1.)

                for key in pol: pol[key] = jnp.nan_to_num(pol[key], 0.)
                pol['w_dopa'] = jnp.array([w_grad, w_hebb, w_ga, 0., 0.])

                # Planning: fixed B_MAX×L_MAX with masking
                if flag_planning:
                    k_plan = random.fold_in(k, s2.info['step'].astype(jnp.int32))
                    acts = random.normal(k_plan, (B_MAX, L_MAX, 8)) * plan_noise
                    def rollout_one(acts_seq):
                        ss = s2; err = 0.
                        for step_i in range(L_MAX):
                            _, pn = policy_forward(pol, ss.obs)
                            ss2 = env.step(ss, acts_seq[step_i]); err += jnp.mean((ss2.obs - pn) ** 2); ss = ss2
                        return err
                    errs = vmap(rollout_one)(acts)
                    a = acts[jnp.argmin(errs), 0]

                # Elite imitation: weight-level alignment (jnp.where gate, JIT-safe)
                elite_cond = (e_fit > mirror_select) & (e_wih.shape[0] > 1)
                align_loss = mirror_lr * jnp.mean((pol['w_ih'] - e_wih) ** 2)
                del_w = jnp.clip(align_loss * (pol['w_ih'] - e_wih) * 0.01, -0.005, 0.005)
                pol['w_ih'] = jnp.where(elite_cond, pol['w_ih'] - del_w, pol['w_ih'])

                return (pol, s2, thought_state), (s2.done, jnp.concatenate([obs_i, a, s2.info['energy'][None]]))
            ts_init = jnp.zeros(TOKEN_DIM) if flag_thought else jnp.zeros(1)
            (pol, s_final, final_ts), (d, ex) = lax.scan(step, (pol, env.reset(k), ts_init), jnp.arange(500))
            d = jnp.nan_to_num(d, nan=1.)
            ex = jnp.nan_to_num(ex, nan=0.)
            fd = jnp.argmax(d > 0.5)
            alive = jnp.where(jnp.any(d > 0.5), fd + 1., 500.)
            final_energy = jnp.nan_to_num(s_final.info['energy'], nan=0.)
            dopa = jnp.nan_to_num(pol['w_dopa'], 0.)
            return (alive, dopa, final_ts), jnp.concatenate([alive[None], final_energy[None], jnp.mean(ex, 0)])
        (alive_arr, dopa_arr, thought_arr), r = vmap(single)(nodes, conns, dopas, regs, spacials, plans, diags, mirrors, thoughts, keys)
        return alive_arr, dopa_arr, thought_arr, r
    return eval_batch

def save_checkpoint(state, ae, gen, curve, path, hf_api=None, hf_repo=None):
    np.savez(path,
        nodes=np.array(state['nodes']), conns=np.array(state['conns']),
        tags=np.array(state['tags']), dopas=np.array(state['dopas']),
        regs=np.array(state['regs']), innov=state['innov'],
        spacials=np.array(state.get('spacials', jnp.zeros(1))),
        plans=np.array(state.get('plans', jnp.zeros(1))),
        diags=np.array(state.get('diags', jnp.zeros(1))),
        mirrors=np.array(state.get('mirrors', jnp.zeros(1))),
        thoughts=np.array(state.get('thoughts', jnp.zeros(1))),
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
    if 'dopas' in d: state['dopas'] = jnp.array(d['dopas'])
    if 'regs' in d: state['regs'] = jnp.array(d['regs'])
    if 'spacials' in d: state['spacials'] = jnp.array(d['spacials'])
    if 'plans' in d: state['plans'] = jnp.array(d['plans'])
    if 'diags' in d: state['diags'] = jnp.array(d['diags'])
    if 'mirrors' in d: state['mirrors'] = jnp.array(d['mirrors'])
    if 'thoughts' in d: state['thoughts'] = jnp.array(d['thoughts'])
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

def run(n_gen=5000, pop_size=1024, seed=3072, resume_path=None, vip_init=None,
        flag_spatial=False, flag_planning=False, flag_diag=False, flag_mirror=False, flag_thought=False):
    key = random.PRNGKey(seed)
    hf_api = None
    try:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            hf_api = HfApi(token=hf_token); print(f"HF OK: {hf_api.whoami()['name']}", flush=True)
    except: pass

    # Elite data for cross-generation imitation
    elite_data = None
    elite_fitness = 0.

    # Init keys
    spatial_key = random.PRNGKey(seed + 10)
    plan_key = random.PRNGKey(seed + 11)
    diag_key = random.PRNGKey(seed + 12)
    mirror_key = random.PRNGKey(seed + 13)
    thought_key = random.PRNGKey(seed + 14)

    # Create eval_batch with flags as closure (JIT-safe)
    eval_batch_fn = make_eval_batch(flag_spatial, flag_planning, flag_diag, flag_mirror, flag_thought)

    if resume_path:
        state, ae, gen_start, curve = load_checkpoint(resume_path)
        pad_genome = lambda key, arr, dim: arr if (arr.shape[1] >= dim and arr.shape[0] > 1) else jnp.zeros((state['pop_size'], dim))
        if 'dopas' not in state or state['dopas'].shape[1] < 5:
            state['dopas'] = init_dopas(random.PRNGKey(seed), state['pop_size'])
        if 'regs' not in state or state['regs'].shape[1] < REG_DIM:
            state['regs'] = init_regs(random.PRNGKey(seed + 2), state['pop_size'])
        if 'spacials' not in state: state['spacials'] = init_spatial(spatial_key, state['pop_size'])
        if 'plans' not in state: state['plans'] = init_plan(plan_key, state['pop_size'])
        if 'diags' not in state: state['diags'] = init_diag(diag_key, state['pop_size'])
        if 'mirrors' not in state: state['mirrors'] = init_mirror(mirror_key, state['pop_size'])
        if 'thoughts' not in state: state['thoughts'] = init_thought(thought_key, state['pop_size'])
        if state['nodes'].shape[-1] < NODE_PARAMS:
            pad = jnp.zeros((state['nodes'].shape[0], MAX_GENES, NODE_PARAMS - state['nodes'].shape[-1]))
            state['nodes'] = jnp.concatenate([state['nodes'], pad], axis=-1)
        print(f"Resumed G{gen_start}, pop={state['pop_size']}", flush=True)
    else:
        ae_key, ga_key = random.split(key)
        if vip_init:
            print(f"VIP init: loading genome from {vip_init}", flush=True)
            vip_gen = load_vip_genome(vip_init)
            k_exp = random.PRNGKey(seed + 42)
            nodes = jnp.tile(vip_gen['nodes'], (pop_size, 1, 1))
            conns = jnp.tile(vip_gen['conns'], (pop_size, 1, 1))
            noise_n = random.normal(k_exp, nodes.shape) * 0.05
            noise_c = random.normal(k_exp, conns.shape) * 0.05
            nodes = nodes + jnp.where(jnp.isnan(nodes), 0., noise_n)
            conns = conns + jnp.where(jnp.isnan(conns), 0., noise_c)
            state = {'nodes': nodes, 'conns': conns, 'tags': jnp.zeros((pop_size, TAG_DIM)),
                     'innov': 6, 'pop_size': pop_size}
        else:
            state = init_pop(ga_key, pop_size)

        state['dopas'] = init_dopas(ga_key, pop_size)
        state['regs'] = init_regs(random.PRNGKey(seed + 66), pop_size)
        state['spacials'] = init_spatial(spatial_key, pop_size)
        state['plans'] = init_plan(plan_key, pop_size)
        state['diags'] = init_diag(diag_key, pop_size)
        state['mirrors'] = init_mirror(mirror_key, pop_size)
        state['thoughts'] = init_thought(thought_key, pop_size)
        ae = init_ae(ae_key)
        gen_start = 0; curve = []

    # Pre-check
    n4 = vmap(lambda i: random.PRNGKey(i))(jnp.arange(4))
    f4, _, _, _ = eval_batch_fn(state['nodes'][:4], state['conns'][:4],
        state['dopas'][:4], state['regs'][:4],
        state['spacials'][:4], state['plans'][:4],
        state['diags'][:4], state['mirrors'][:4], state['thoughts'][:4], n4)
    flags_str = f" spatial={flag_spatial} planning={flag_planning} diag={flag_diag} mirror={flag_mirror} thought={flag_thought}"
    print(f"V2.9.x vip={vip_init is not None}: {n_gen}gen x {pop_size}pop{flags_str}, pre-check fitness={float(jnp.max(f4)):.0f}", flush=True)

    t0 = time.time()
    k0 = random.PRNGKey(gen_start * 3)
    for g in range(gen_start, n_gen):
        k0 = random.PRNGKey(g * 3)
        fs, dopas, regss, rss, thoughtss = [], [], [], [], []
        for ri in range(1):
            kk = random.split(random.fold_in(k0, ri), pop_size)
            f_arr, d_arr, th_arr, r = eval_batch_fn(state['nodes'], state['conns'],
                state['dopas'], state['regs'],
                state['spacials'], state['plans'],
                state['diags'], state['mirrors'], state['thoughts'], kk,
                elite_data=elite_data if elite_data is not None else (jnp.zeros(1), 0.))
            fs.append(f_arr); dopas.append(d_arr); rss.append(r)
            thoughtss.append(th_arr)
            regss.append(jnp.mean(state['regs'], axis=0))
        f = jnp.nan_to_num(f_arr, nan=0.)
        dopa_mean = jnp.nan_to_num(jnp.mean(jnp.stack(dopas), axis=(0,1)), nan=0.)
        regs_mean = jnp.nan_to_num(jnp.mean(jnp.stack(regss), axis=0), nan=0.)
        r = rss[0]
        # AE input = [actions(8), fitness/500, energy(1)]
        # Actions are always the 8 values before the last (energy)
        ae_input = jnp.concatenate([
            r[:, -9:-1], f[:, None] / 500., r[:, -1:]], axis=1)

        ae = train_ae(ae, ae_input, random.PRNGKey(g + 1000))
        dm_ae = jnp.mean(ae_input); ds_ae = jnp.std(ae_input) + 1e-3
        ae_norm = jnp.clip((ae_input - dm_ae) / ds_ae, -3., 3.)
        nt = jnp.nan_to_num(vmap(lambda e: encode(ae, e))(ae_norm), nan=0.)
        all_dec = vmap(lambda e: decode(ae, encode(ae, e)))(ae_norm)
        per_loss = jnp.mean((ae_norm - all_dec) ** 2, axis=1)
        dopamine = (per_loss - jnp.min(per_loss)) / (jnp.max(per_loss) - jnp.min(per_loss) + 1e-8)
        ae_loss = jnp.mean(per_loss)
        f_total = f + 5 * dopamine
        curve.append((float(jnp.max(f_total)), float(jnp.mean(f_total))))

        # Elite-based imitation: update elite if best agent improved
        best_idx = jnp.argmax(f_total)
        if flag_mirror and f_total[best_idx] > elite_fitness:
            elite_fitness = float(f_total[best_idx])
            bi = best_idx
            pol_e = genome_to_policy(state['nodes'][bi], state['conns'][bi], regs=state['regs'][bi])
            elite_data = (pol_e['w_ih'], elite_fitness)

        # Self-diagnosis: adjust mutation rate (per-gen: use gen 0 params as scalar)
        if flag_diag:
            diag_0 = state['diags'][0]  # shape (3,) — use first agent's params as representative
            monitor_window = int(50 + 150 * float(jax.nn.sigmoid(diag_0[0])))
            anomaly_threshold = 1.5 + 1.5 * float(jax.nn.sigmoid(diag_0[1]))
            reflect_interval = int(50 + 450 * float(jax.nn.sigmoid(diag_0[2])))
            if g % reflect_interval == 0 and g > 0:
                # compare current ae_loss to historical mean
                hist_mean = jnp.mean(jnp.array([c[1] for c in curve[-monitor_window:]]))
                if ae_loss > hist_mean + anomaly_threshold * hist_mean:
                    mr = float(jnp.clip(dopa_mean[2], 0.05, 1.0)) * 1.2  # explore
                elif ae_loss < hist_mean - anomaly_threshold * hist_mean:
                    mr = float(jnp.clip(dopa_mean[2], 0.05, 1.0)) * 0.9  # exploit
                else:
                    mr = float(jnp.clip(dopa_mean[2], 0.05, 1.0))
            else:
                mr = float(jnp.clip(dopa_mean[2], 0.05, 1.0))
        else:
            mr = float(jnp.clip(dopa_mean[2], 0.05, 1.0))

        w_ga = mr
        k1, k2 = random.split(random.PRNGKey(g + seed))
        idx = random.randint(k1, (pop_size, 3), 0, pop_size)
        pr = idx[jnp.arange(pop_size), jnp.argmax(f_total[idx], axis=1)]
        np2 = pop_size // 2
        p1_arr = pr[0::2]; p2_arr = pr[1::2]

        # Vmap all crossover keys
        kx_arr = vmap(lambda i: random.fold_in(k2, i * 7 + 0))(jnp.arange(np2))
        ky_arr = vmap(lambda i: random.fold_in(k2, i * 7 + 1))(jnp.arange(np2))
        kt_arr = vmap(lambda i: random.fold_in(k2, i * 7 + 2))(jnp.arange(np2))
        ks_arr = vmap(lambda i: random.fold_in(k2, i * 7 + 3))(jnp.arange(np2))
        kr_arr = vmap(lambda i: random.fold_in(k2, i * 7 + 4))(jnp.arange(np2))
        ksp_arr = vmap(lambda i: random.fold_in(k2, i * 7 + 5))(jnp.arange(np2))
        kpl_arr = vmap(lambda i: random.fold_in(k2, i * 7 + 6))(jnp.arange(np2))

        # Main genome crossover
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

        # Tag crossover (Lamarkian)
        ct_batch = vmap(crossover_tags)
        tc = tc.at[0::2].set(ct_batch(nt[p1_arr], nt[p2_arr], kt_arr))
        tc = tc.at[1::2].set(ct_batch(nt[p2_arr], nt[p1_arr], ks_arr))

        # Dopamine crossover
        cd_batch = vmap(crossover_dopas)
        dc = jnp.zeros((pop_size, 5))
        dc = dc.at[0::2].set(cd_batch(state['dopas'][p1_arr], state['dopas'][p2_arr], kt_arr))
        dc = dc.at[1::2].set(cd_batch(state['dopas'][p2_arr], state['dopas'][p1_arr], ks_arr))

        # Regulatory crossover
        cr_batch = vmap(crossover_regs)
        rc = jnp.zeros((pop_size, REG_DIM))
        rc = rc.at[0::2].set(cr_batch(state['regs'][p1_arr], state['regs'][p2_arr], kt_arr))
        rc = rc.at[1::2].set(cr_batch(state['regs'][p2_arr], state['regs'][p1_arr], ks_arr))

        # Spatial, planning, diag, mirror crossover (always propagate even if flag=0)
        cs_batch = vmap(crossover_spatial)
        sc = jnp.zeros((pop_size, SPATIAL_DIM))
        sc = sc.at[0::2].set(cs_batch(state['spacials'][p1_arr], state['spacials'][p2_arr], ksp_arr))
        sc = sc.at[1::2].set(cs_batch(state['spacials'][p2_arr], state['spacials'][p1_arr], kpl_arr))
        cp_batch = vmap(crossover_plan)
        pc = jnp.zeros((pop_size, PLAN_DIM))
        pc = pc.at[0::2].set(cp_batch(state['plans'][p1_arr], state['plans'][p2_arr], ksp_arr))
        pc = pc.at[1::2].set(cp_batch(state['plans'][p2_arr], state['plans'][p1_arr], kpl_arr))
        cdg_batch = vmap(crossover_diag)
        dgc = jnp.zeros((pop_size, DIAG_DIM))
        dgc = dgc.at[0::2].set(cdg_batch(state['diags'][p1_arr], state['diags'][p2_arr], ksp_arr))
        dgc = dgc.at[1::2].set(cdg_batch(state['diags'][p2_arr], state['diags'][p1_arr], kpl_arr))
        cm_batch = vmap(crossover_mirror)
        mc = jnp.zeros((pop_size, MIRROR_DIM))
        mc = mc.at[0::2].set(cm_batch(state['mirrors'][p1_arr], state['mirrors'][p2_arr], ksp_arr))
        mc = mc.at[1::2].set(cm_batch(state['mirrors'][p2_arr], state['mirrors'][p1_arr], kpl_arr))
        # Thought crossover
        cth_batch = vmap(crossover_thought)
        thc = jnp.zeros((pop_size, THOUGHT_DIM))
        thc = thc.at[0::2].set(cth_batch(state['thoughts'][p1_arr], state['thoughts'][p2_arr], ksp_arr))
        thc = thc.at[1::2].set(cth_batch(state['thoughts'][p2_arr], state['thoughts'][p1_arr], kpl_arr))

        # Lamarkian tag delta
        td_arr = vmap(lambda p1, p2: jnp.tile(nt[p1] - nt[p2], 7)[:MAX_GENES])(p1_arr, p2_arr)
        delta = jnp.zeros((np2, MAX_GENES, NODE_PARAMS)).at[:, :, 5].set(0.01 * td_arr)
        cn = cn.at[0::2].set(cn[0::2] + delta)
        cn = cn.at[1::2].set(cn[1::2] + delta)

        # Mutate all genomes
        cn, cc, ni_jit = mutate(cn, cc, k2, state['innov'], .1*mr, .03*mr, .02*mr)
        ni = int(ni_jit)
        tc = mutate_tags(tc, k2, noise=.03*mr)
        dc = mutate_dopas(dc, k2, noise=.03*mr)
        rc = mutate_regs(rc, k2, noise=.03*mr)
        sc = mutate_spatial(sc, k2, noise=.03*mr)
        pc = mutate_plan(pc, k2, noise=.03*mr)
        dgc = mutate_diag(dgc, k2, noise=.03*mr)
        mc = mutate_mirror(mc, k2, noise=.03*mr)
        thc = mutate_thought(thc, k2, noise=.03*mr)

        # Elitism top 2
        top2 = jnp.argsort(f_total)[-2:]
        for j in range(2):
            cn = cn.at[j].set(state['nodes'][top2[j]]); cc = cc.at[j].set(state['conns'][top2[j]])
            tc = tc.at[j].set(state['tags'][top2[j]]); dc = dc.at[j].set(state['dopas'][top2[j]])
            rc = rc.at[j].set(state['regs'][top2[j]]); sc = sc.at[j].set(state['spacials'][top2[j]])
            pc = pc.at[j].set(state['plans'][top2[j]]); dgc = dgc.at[j].set(state['diags'][top2[j]])
            mc = mc.at[j].set(state['mirrors'][top2[j]]); thc = thc.at[j].set(state['thoughts'][top2[j]])

        state['nodes'] = cn; state['conns'] = cc; state['tags'] = tc
        state['dopas'] = dc; state['regs'] = rc
        state['spacials'] = sc; state['plans'] = pc
        state['diags'] = dgc; state['mirrors'] = mc; state['thoughts'] = thc
        state['innov'] = ni

        if (g + 1) % 20 == 0:
            dt = time.time() - t0; eta = dt/(g-gen_start+1)*(n_gen-g-1)/60
            n_active = int(jnp.sum(jax.nn.sigmoid(state['regs'][:4, :8]) > 0.5).mean())
            print(f"G{g+1}: max={curve[-1][0]:.0f} mean={curve[-1][1]:.0f}"
                  f" rmx={float(jnp.max(f)):.0f} rmn={float(jnp.mean(f)):.0f}"
                  f" ae={ae_loss:.4f} dopa={dopa_mean[0]:.2f}/{dopa_mean[1]:.2f}/{dopa_mean[2]:.2f}"
                  f" mod={n_active} mr={mr:.2f}"
                  f" elite={elite_fitness:.0f}"
                  f" [{dt:.0f}s ETA {eta:.0f}m]", flush=True)

        if (g + 1) % 500 == 0 or g == n_gen - 1:
            os.makedirs("checkpoints/v2.9", exist_ok=True)
            cp_path = f"checkpoints/v2.9/cp_{g+1}.npz"
            save_checkpoint(state, ae, g+1, curve, cp_path, hf_api, "hhian/checkpoints")

    ffs = []
    for ri in range(1):
        kk = random.split(random.fold_in(k0, ri), pop_size)
        ff, _, _, _ = eval_batch_fn(state['nodes'], state['conns'],
            state['dopas'], state['regs'],
            state['spacials'], state['plans'],
            state['diags'], state['mirrors'], state['thoughts'], kk,
            elite_data=elite_data if elite_data is not None else (jnp.zeros(1), 0.))
        ffs.append(ff)
    ff = jnp.nan_to_num(ff, nan=0.)
    bi = int(jnp.argmax(ff))
    print(f"Best steps (raw): {float(jnp.max(ff)):.0f}", flush=True)
    os.makedirs("v2_6/results", exist_ok=True)
    ts = time.strftime("%m%d_%H%M")
    np.savez(f'v2_6/results/v291_{ts}.npz',
        curve=np.array(curve), best_nodes=np.array(state['nodes'][bi]),
        best_conns=np.array(state['conns'][bi]),
        best_regs=np.array(state['regs'][bi]),
        best_spatials=np.array(state['spacials'][bi]) if flag_spatial else np.zeros(1),
        best_plans=np.array(state['plans'][bi]) if flag_planning else np.zeros(1),
        best_diags=np.array(state['diags'][bi]) if flag_diag else np.zeros(1),
        best_mirrors=np.array(state['mirrors'][bi]) if flag_mirror else np.zeros(1),
        best_fitness=float(jnp.max(ff)),
        best_total_fitness=float(curve[-1][0]))
    return {'curve': jnp.array(curve), 'best_nodes': state['nodes'][bi],
            'best_conns': state['conns'][bi], 'best_fitness': float(jnp.max(ff))}

if __name__ == "__main__":
    import sys
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    pop_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 3072
    mode = sys.argv[4] if len(sys.argv) > 4 else "train"
    args = sys.argv[5:]

    flag_spatial = '--spatial' in args
    flag_planning = '--planning' in args
    flag_diag = '--diagnosis' in args
    flag_mirror = '--imitation' in args
    flag_thought = '--thought' in args

    if mode == "teacher":
        print("Training teacher (gradient + curiosity)...")
        teacher = train_teacher(n_episodes=200, lr=0.001, seed=seed)
        print("Compressing teacher → genome...")
        genome = compress_teacher(teacher, n_opt_steps=500, seed=seed + 100)
        save_vip_genome(genome)
    elif mode.startswith("vip"):
        vip_path = sys.argv[5] if len(sys.argv) > 5 and not sys.argv[5].startswith('--') else "vip_genome.npz"
        run(n_gen=n_gen, pop_size=pop_size, seed=seed, vip_init=vip_path,
            flag_spatial=flag_spatial, flag_planning=flag_planning,
            flag_diag=flag_diag, flag_mirror=flag_mirror, flag_thought=flag_thought)
    elif mode == "resume":
        resume_path = sys.argv[5] if len(sys.argv) > 5 and not sys.argv[5].startswith('--') else None
        run(n_gen=n_gen, pop_size=pop_size, seed=seed, resume_path=resume_path,
            flag_spatial=flag_spatial, flag_planning=flag_planning,
            flag_diag=flag_diag, flag_mirror=flag_mirror, flag_thought=flag_thought)
    else:
        run(n_gen=n_gen, pop_size=pop_size, seed=seed,
            flag_spatial=flag_spatial, flag_planning=flag_planning,
            flag_diag=flag_diag, flag_mirror=flag_mirror, flag_thought=flag_thought)