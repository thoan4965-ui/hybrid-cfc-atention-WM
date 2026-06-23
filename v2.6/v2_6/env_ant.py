"""NoRewardAnt — 3 rings, 6 foods total, NO respawn, spawn at origin."""
import jax
import jax.numpy as jnp
from brax.envs import ant

class NoRewardAnt(ant.Ant):
    def __init__(self, energy_init=50., energy_cost=0.4, torque_cost=0.05,
                 food_energy=50, arena=20., **kw):
        super().__init__(**kw)
        self._e_init = energy_init; self._e_cost = energy_cost
        self._t_cost = torque_cost; self._f_energy = food_energy
        self._arena = arena
        self._n_rings = 3; self._n_per_ring = 2
        self._n_food = self._n_rings * self._n_per_ring
        self._radii = jnp.array([5., 10., 15.])

    def reset(self, rng):
        s = super().reset(rng)
        ks = jax.random.split(rng, self._n_rings + 1)
        ring_keys = ks[:self._n_rings]
        def make_ring(k, r):
            ang = jax.random.uniform(k, (self._n_per_ring,)) * 2 * jnp.pi
            return jnp.stack([jnp.cos(ang)*r, jnp.sin(ang)*r], 1)
        fp = jnp.concatenate([make_ring(ring_keys[i], self._radii[i])
                              for i in range(self._n_rings)])
        s.info.update({'energy': jnp.full((), self._e_init),
                       'food_pos': fp,
                       'food_cnt': jnp.full((self._n_food,), 1.),
                       'step': jnp.full((), 0.)})
        s = s.replace(obs=jnp.concatenate([s.obs, jnp.zeros(2)]))
        return s

    def step(self, state, action):
        ps = self.pipeline_step(state.pipeline_state, action)
        e = state.info['energy'] - self._e_cost - self._t_cost * jnp.sum(jnp.square(action))
        st = state.info['step'] + 1.
        ap = ps.x.pos[0,:2]; fp = state.info['food_pos']
        fc = state.info['food_cnt']
        d = jnp.sqrt(jnp.sum((fp - ap)**2, 1))
        eaten = (d < 1.) & (fc > 0)
        e += jnp.sum(eaten) * self._f_energy
        fc = jnp.where(eaten, fc - 1., fc)
        healthy = jnp.where(ps.x.pos[0,2] > 0.2, 1., 0.)
        has_e = jnp.where(e > 0, 1., 0.)
        done = 1. - (healthy * has_e)
        obs = self._get_obs(ps)
        has_food = jnp.any(fc > 0)
        nearest = jnp.argmin(d + (fc <= 0) * 999., axis=0)
        dx = jnp.where(has_food, fp[nearest, 0] - ap[0], 0.)
        dy = jnp.where(has_food, fp[nearest, 1] - ap[1], 0.)
        obs = jnp.concatenate([obs, jnp.array([dx, dy])])
        return state.replace(pipeline_state=ps, obs=obs,
                             reward=jnp.zeros_like(state.reward), done=done,
                             info={'energy': e, 'food_pos': fp,
                                   'food_cnt': fc, 'step': st})
