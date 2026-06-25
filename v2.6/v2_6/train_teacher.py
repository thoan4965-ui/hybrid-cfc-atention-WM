"""Teacher: gradient + curiosity. JIT-compiled rollout via lax.scan."""
import jax, jax.numpy as jnp
jax.config.update('jax_default_matmul_precision', 'high')
from jax import random, jit, lax
from v2_6.cppn import policy_forward
from v2_6.env_ant import NoRewardAnt

env = NoRewardAnt(backend='mjx', energy_init=20., energy_cost=0.4, torque_cost=0.05)
if hasattr(env, 'sys') and hasattr(env.sys, 'mj_model'):
    env.sys.mj_model.opt.iterations = 3
    env.sys.mj_model.opt.ls_iterations = 5
    env.sys.mj_model.opt.timestep = 0.001

def init_teacher(key):
    k1, k2, k3 = random.split(key, 3)
    return {
        'w_ih': random.normal(k1, (30, 10)) * 0.1,
        'w_ho': random.normal(k2, (10, 8)) * 0.1,
        'w_pred': random.normal(k3, (10, 29)) * 0.1,
    }

@jit
def teacher_loss(params, obs_seq, action_seq, next_obs_seq):
    h = jnp.tanh(obs_seq @ params['w_ih'][:-1] + params['w_ih'][-1])
    pred_next = h @ params['w_pred']
    pred_err = jnp.mean((next_obs_seq - pred_next) ** 2)
    curiosity = -0.01 * (jnp.var(jnp.tanh(h @ params['w_ho'])) + jnp.var(action_seq))
    return pred_err + curiosity

@jit
def rollout(key, params, n_steps=200):
    """JIT-compiled rollout. Returns full arrays + alive count."""
    def step(s, _):
        a, pred_n = policy_forward(params, s.obs)
        s2 = env.step(s, a)
        s2 = s2.replace(obs=jnp.nan_to_num(s2.obs, 0.))
        return s2, (s.obs, a, s2.obs, s2.done)
    init_state = env.reset(key)
    _, (obs, act, nxt, dones) = lax.scan(step, init_state, jnp.arange(n_steps))
    fd = jnp.argmax(dones > 0.5)
    alive = jnp.where(jnp.any(dones > 0.5), fd + 1, n_steps)
    return obs, act, nxt, alive

def train_teacher(n_episodes=500, lr=0.001, seed=3072):
    """Train teacher policy with gradient + curiosity."""
    key = random.PRNGKey(seed)
    params = init_teacher(key)

    # JIT compile rollout once
    _, _, _, _ = rollout(random.fold_in(key, 0), params)

    for ep in range(n_episodes):
        k_ep = random.fold_in(key, ep)
        obs, act, nxt, alive = rollout(k_ep, params)
        loss, grads = jax.value_and_grad(
            lambda p: teacher_loss(p, obs, act, nxt))(params)
        params = {k: params[k] - lr * jnp.clip(grads[k], -0.5, 0.5)
                  for k in params}
        if (ep + 1) % 20 == 0:
            _, _, _, alive_e = rollout(random.fold_in(key, ep + 9999), params)
            avg_steps = float(alive_e)
            print(f"  Teacher ep{ep+1}: loss={float(loss):.4f} avg_steps={avg_steps:.0f}")
    return params
