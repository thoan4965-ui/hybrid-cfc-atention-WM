"""NoRewardAnt — Brax Ant with reward=0, energy depletion, food respawn."""
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs import ant

class NoRewardAnt(ant.Ant):
    def __init__(self, energy_init=1000., energy_cost=0.01, torque_cost=0.01,
                 n_food=8, food_units=10, food_energy=300,
                 cd=50, arena=20., **kw):
        super().__init__(**kw)
        self._e_init = energy_init; self._e_cost = energy_cost
        self._t_cost = torque_cost; self._n_food = n_food
        self._f_units = food_units; self._f_energy = food_energy
        self._cd = cd; self._arena = arena

    def reset(self, rng):
        s = super().reset(rng)
        k, sk = jax.random.split(rng)
        fp = jax.random.uniform(sk, (self._n_food,2), minval=-self._arena/2, maxval=self._arena/2)
        s.info.update({'energy': jnp.full((), self._e_init),
                       'food_pos': fp, 'food_cnt': jnp.full((self._n_food,), float(self._f_units)),
                       'food_cd': jnp.full((self._n_food,), -1.), 'step': jnp.full((), 0.)})
        return s

    def step(self, state, action):
        ps = self.pipeline_step(state.pipeline_state, action)
        e = state.info['energy'] - self._e_cost - self._t_cost * jnp.sum(jnp.square(action))
        st = state.info['step'] + 1.
        ap = ps.x.pos[0,:2]; fp = state.info['food_pos']
        fc = state.info['food_cnt']; fcd = state.info['food_cd']
        d = jnp.sqrt(jnp.sum((fp - ap)**2, 1))
        eaten = (d < 1.) & (fc > 0) & (fcd < 0)
        e += jnp.sum(eaten) * self._f_energy
        fc = jnp.where(eaten, fc - 1., fc)
        fcd = jnp.where((fc <= 0) & (fcd < 0), float(self._cd), fcd)
        fcd = jnp.maximum(fcd - 1., -1.)
        resp = (fc <= 0) & (fcd >= 0) & (fcd < 0.5)
        nx = jnp.sin(st*0.1 + jnp.arange(self._n_food)) * self._arena/2
        ny = jnp.cos(st*0.13 + jnp.arange(self._n_food)) * self._arena/2
        fp = jnp.where(resp[:,None], jnp.stack([nx,ny],1), fp)
        fc = jnp.where(resp, float(self._f_units), fc)
        fcd = jnp.where(resp, -1., fcd)
        healthy = jnp.where(ps.x.pos[0,2] > 0.2, 1., 0.)
        has_e = jnp.where(e > 0, 1., 0.)
        done = 1. - (healthy * has_e)
        obs = self._get_obs(ps)  # base 27-dim, ko append energy
        return state.replace(pipeline_state=ps, obs=obs,
                             reward=jnp.zeros_like(state.reward), done=done,
                             info={'energy': e, 'food_pos': fp, 'food_cnt': fc,
                                   'food_cd': fcd, 'step': st})
