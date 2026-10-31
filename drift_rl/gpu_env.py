"""
gpu_env.py - the drift environment with every car simulated on an NVIDIA GPU.

One GPU thread per car runs drift_env._step_one - the same compiled-from-the-same-source
function the CPU environment loops over (see sim_target.py). All state lives in GPU memory as
PyTorch tensors; numba's CUDA kernels write into those same tensors, so observations go
straight from the simulator into the policy network without ever visiting the CPU.

Needs: an NVIDIA GPU, PyTorch built with CUDA, and numba-cuda (see README, "GPU setup").

    env = DriftGpuEnv(world, n_envs=4096)
    obs = env.reset()                         # torch tensor (n_envs, 46), float64, on the GPU
    obs, reward, terminated, truncated = env.step(actions)

The tensors returned are live views that the next step() overwrites - clone what you keep.

TorchCpuEnv gives the CPU simulator the same interface, so train_ppo_gpu.py runs on either.
"""
import math

import numpy as np

import sim_target
import fast_sim as fs_cpu
import drift_env as de_cpu


def kernel_arg_names():
    """Argument order of drift_env.step_kernel / step_batch, in one place."""
    return (['states', 'mids', 'diags', 'actions', 'prev_act', 'counters', 'ep', 'params', 'rw', 'obs',
             'rewards', 'terminated', 'truncated', 'final_obs', 'ep_out', 'rnd', 'env_cfg',
             'episode_ticks', 'sp_dir', 'la_idx', 'la_min_radius', 'pace_ref'], _WORLD_NAMES + ['gh', 'gh_len'])


_WORLD_NAMES = ['sp_pos', 'sp_radius', 'sp_length', 'sp_next', 'sp_prev', 'sp_fwd', 'sp_cum', 'tri_v',
                'tri_surf', 'gmin', 'cell_size', 'dims', 'cell_start', 'cell_items', 'rm_pos', 'rm_min', 'rm_cell',
                'rm_dims', 'rm_start', 'rm_items', 'co', 'mk']
_RESET_NAMES = ['states', 'mids', 'prev_act', 'counters', 'ep', 'params', 'obs', 'rnd', 'env_cfg',
                'sp_dir', 'la_idx', 'la_min_radius', 'sp_pos', 'sp_radius', 'sp_length', 'sp_next',
                'sp_prev', 'sp_fwd', 'co']


def host_arrays(world, n_envs, env_cfg, rew_cfg):
    """Every array the kernels take, as numpy, keyed by argument name. Used to build the GPU
    environment and (by check_gpu.py) to run the CPU and GPU code on identical inputs."""
    sp_dir, la_idx, la_min_radius = de_cpu.spline_lookahead_tables(world)
    base = fs_cpu.make_params()
    de_cpu.apply_plugin_settings(base, env_cfg.plugin_settings)
    rw = rew_cfg.vector()
    if rw[de_cpu.R_TRACK_SURF] < 0:
        rw[de_cpu.R_TRACK_SURF] = de_cpu.track_surface_code(world)
    a = {
        'states': np.zeros((n_envs, fs_cpu.STATE_SIZE)), 'mids': np.zeros((n_envs, fs_cpu.MID_SIZE)),
        'diags': np.zeros((n_envs, fs_cpu.DIAG_SIZE)), 'actions': np.zeros((n_envs, de_cpu.ACT_SIZE)),
        'prev_act': np.zeros((n_envs, de_cpu.ACT_SIZE)), 'counters': np.zeros((n_envs, de_cpu.N_COUNTERS)),
        'ep': np.zeros((n_envs, de_cpu.EP_SIZE)), 'params': np.tile(base, (n_envs, 1)), 'rw': rw,
        'obs': np.zeros((n_envs, de_cpu.OBS_SIZE)), 'rewards': np.zeros(n_envs),
        'terminated': np.zeros(n_envs, dtype=np.bool_), 'truncated': np.zeros(n_envs, dtype=np.bool_),
        'final_obs': np.zeros((n_envs, de_cpu.OBS_SIZE)), 'ep_out': np.zeros((n_envs, de_cpu.EP_SIZE)),
        'rnd': np.zeros((n_envs, de_cpu.RND_COLS)),
        'env_cfg': de_cpu.env_cfg_vector(env_cfg),
        'episode_ticks': float(env_cfg.episode_ticks),
        'sp_dir': sp_dir, 'la_idx': la_idx, 'la_min_radius': la_min_radius,
        # the course's reference speed per point for paces (drift_env.pace_reference); zeros = none
        'pace_ref': np.ascontiguousarray(getattr(world, 'pace_ref', None) if getattr(world, 'pace_ref', None) is not None
                                         else np.zeros(len(world.sp_pos)), dtype=np.float64),
    }
    for name, arr in zip(_WORLD_NAMES, world.arrays()):
        a[name] = arr
    # ghost leaders (chase episodes): passed to the step, but the GPU environment starts no chases yet
    a['gh'], a['gh_len'] = de_cpu.ghost_arrays(world)
    a['_base_params'] = base
    return a


class DriftGpuEnv:
    def __init__(self, world, n_envs=4096, env_config=None, reward_config=None, seed=0,
                 threads_per_block=64):
        import os
        import torch
        from numba import cuda
        # NUMBA_ENABLE_CUDASIM=1 is numba's no-GPU test mode: the kernels run as plain Python on
        # CPU tensors. Far too slow to train with; it exists so this class can be exercised
        # end to end on a machine without an NVIDIA card.
        simulated = os.environ.get('NUMBA_ENABLE_CUDASIM') == '1'
        if simulated:
            self.device = torch.device('cpu')
            as_kernel_array = lambda t: t.numpy()
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch cannot see a CUDA GPU (torch.cuda.is_available() is False). "
                                   "Install the CUDA build of PyTorch - see README, 'GPU setup'.")
            torch.cuda.init()
            self.device = torch.device('cuda')
            as_kernel_array = cuda.as_cuda_array
        self.torch = torch
        self.n = n_envs
        self.cfg = env_config or de_cpu.EnvConfig()
        self.rcfg = reward_config or de_cpu.RewardConfig()
        self.de = sim_target.load_variant('drift_env', 'cuda')
        host = host_arrays(world, n_envs, self.cfg, self.rcfg)
        self.base_params = host.pop('_base_params')
        self.t = {}       # name -> torch tensor on the GPU
        self.d = {}       # name -> numba view of the same memory (or a plain scalar)
        for name, v in host.items():
            if isinstance(v, np.ndarray):
                self.t[name] = torch.from_numpy(np.ascontiguousarray(v)).to(self.device)
                self.d[name] = as_kernel_array(self.t[name])
            else:
                self.d[name] = v
        names, world_names = kernel_arg_names()
        self._step_args = tuple(self.d[k] for k in names + world_names)
        self._reset_args = tuple(self.d[k] for k in _RESET_NAMES)
        self._tpb = threads_per_block
        self._blocks = (n_envs + threads_per_block - 1) // threads_per_block
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.obs, self.rewards = self.t['obs'], self.t['rewards']
        self.terminated, self.truncated = self.t['terminated'], self.t['truncated']
        self.final_obs, self.ep_out = self.t['final_obs'], self.t['ep_out']
        self.diags, self.states = self.t['diags'], self.t['states']
        self._grip_idx = [fs_cpu.P['LAT_FRONT_PEAK_FRICTION'], fs_cpu.P['LAT_REAR_PEAK_FRICTION']]

    def _fill_random(self):
        torch = self.torch
        self.t['rnd'].copy_(torch.rand(self.t['rnd'].shape, generator=self.gen, device=self.device,
                                       dtype=torch.float64))

    def _randomise_grip(self, mask):
        torch = self.torch
        k = int(mask.sum())
        if k == 0:
            return
        for share, names in ((self.cfg.dr_axle_grip, ('DR_FRONT_GRIP_SCALE', 'DR_REAR_GRIP_SCALE')),
                             (self.cfg.dr_peak_slip, ('DR_FRONT_PEAK_SLIP_SCALE', 'DR_REAR_PEAK_SLIP_SCALE'))):
            if share > 0.0:
                for name in names:
                    self.t['params'][mask, fs_cpu.P[name]] = \
                        1.0 + (torch.rand(k, generator=self.gen, device=self.device, dtype=torch.float64) * 2 - 1) * share
        g = self.cfg.grip_randomisation
        if g <= 0.0:
            return
        scale = 1.0 + (torch.rand(k, generator=self.gen, device=self.device, dtype=torch.float64) * 2 - 1) * g
        for idx in self._grip_idx:
            self.t['params'][mask, idx] = self.base_params[idx] * scale

    def set_sliding_grip_low(self, low):
        """Bottom of the range new episodes draw RearSlidingGrip from (the trainer walks it down)."""
        self.t['env_cfg'][de_cpu.ENV_CFG_SLIDING_GRIP_LOW] = low

    def set_pace_low(self, low):
        """Bottom of the range paces are drawn from (the trainer walks it down)."""
        self.t['env_cfg'][de_cpu.ENV_CFG_PACE_LOW] = low

    def set_chase_gap_low(self, low):
        """No chase episodes on the GPU simulator."""

    def reset(self):
        self._randomise_grip(self.torch.ones(self.n, dtype=self.torch.bool, device=self.device))
        self._fill_random()
        self.de.reset_kernel[self._blocks, self._tpb](*self._reset_args)
        return self.obs

    def step(self, actions):
        if actions.shape[1] < de_cpu.ACT_SIZE:
            # a policy with fewer outputs (two: no brake action) - the rest are 0
            self.t['actions'].zero_()
            self.t['actions'][:, :actions.shape[1]].copy_(actions)
        else:
            self.t['actions'].copy_(actions)
        self._fill_random()
        self.de.step_kernel[self._blocks, self._tpb](*self._step_args)
        if self.cfg.grip_randomisation > 0.0 or self.cfg.dr_axle_grip > 0.0 or self.cfg.dr_peak_slip > 0.0:
            self._randomise_grip(self.terminated | self.truncated)
        return self.obs, self.rewards, self.terminated, self.truncated


class TorchCpuEnv:
    """The CPU simulator (drift_env.DriftVecEnv, all cores) behind the same interface."""

    def __init__(self, world, n_envs=256, env_config=None, reward_config=None, seed=0):
        import torch
        self.torch = torch
        self.device = torch.device('cpu')
        self.n = n_envs
        self.env = de_cpu.DriftVecEnv(world, n_envs, env_config, reward_config, seed=seed)
        self.cfg, self.rcfg = self.env.cfg, self.env.rcfg
        e = self.env
        self.obs, self.rewards = torch.from_numpy(e.obs), torch.from_numpy(e.rewards)
        self.terminated, self.truncated = torch.from_numpy(e.terminated), torch.from_numpy(e.truncated)
        self.final_obs, self.ep_out = torch.from_numpy(e.final_obs), torch.from_numpy(e.ep_out)
        self.diags, self.states = torch.from_numpy(e.diags), torch.from_numpy(e.states)
        # the privileged critic's inputs (EnvConfig.privileged; drift_env.PRIV_NAMES)
        self.priv, self.final_priv = torch.from_numpy(e.priv), torch.from_numpy(e.final_priv)

    def set_sliding_grip_low(self, low):
        """Bottom of the range new episodes draw RearSlidingGrip from (the trainer walks it down)."""
        self.env.env_cfg[de_cpu.ENV_CFG_SLIDING_GRIP_LOW] = low

    def set_pace_low(self, low):
        """Bottom of the range paces are drawn from (the trainer walks it down)."""
        self.env.env_cfg[de_cpu.ENV_CFG_PACE_LOW] = low

    def set_chase_gap_low(self, low):
        """Bottom of the range chase episodes draw the asked gap from (the trainer walks it down)."""
        self.env.cfg.chase_gap_low = low

    def reset(self):
        self.env.reset()
        return self.obs

    def step(self, actions):
        self.env.step(actions.detach().cpu().numpy().astype(np.float64))
        return self.obs, self.rewards, self.terminated, self.truncated


class MultiTrackCpuEnv:
    """Several tracks at once on the CPU simulator: the cars in groups, each group a TorchCpuEnv on
    one track, behind the same interface (the observations etc. of all groups, end to end). The
    fixed worlds keep a group each for the whole run; the other groups are dealt worlds from the
    pool and dealt new ones by rotate()."""

    def __init__(self, fixed, pool, n_envs, group, env_config=None, reward_config=None, seed=0):
        import torch
        self.torch = torch
        self.device = torch.device('cpu')
        self.n = n_envs
        self.fixed, self.pool, self.group = list(fixed), list(pool), group
        self.n_groups = n_envs // group
        if self.n_groups * group != n_envs or self.n_groups < len(self.fixed) + (1 if self.pool else 0):
            raise ValueError(f"{n_envs} cars do not split into groups of {group} covering "
                             f"{len(self.fixed)} fixed tracks{' and the pool' if self.pool else ''}")
        self.env_config, self.reward_config = env_config, reward_config
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.grip_low = None
        self.pace_low = None
        self.groups = [None] * self.n_groups
        self.world_of = [None] * self.n_groups
        for g, w in enumerate(self.fixed):
            self._deal(g, w)
        for g in range(len(self.fixed), self.n_groups):
            self._deal(g, self.pool[self.rng.integers(len(self.pool))] if self.pool else self.fixed[g % len(self.fixed)])
        self.cfg, self.rcfg = self.groups[0].cfg, self.groups[0].rcfg
        self._gather()

    def _deal(self, g, world):
        self.seed += 1
        env = TorchCpuEnv(world, self.group, self.env_config, self.reward_config, seed=self.seed)
        if self.grip_low is not None:
            env.set_sliding_grip_low(self.grip_low)
        if self.pace_low is not None:
            env.set_pace_low(self.pace_low)
        self.groups[g], self.world_of[g] = env, world

    def _gather(self):
        cat = self.torch.cat
        G = self.groups
        self.obs = cat([e.obs for e in G])
        self.rewards = cat([e.rewards for e in G])
        self.terminated = cat([e.terminated for e in G])
        self.truncated = cat([e.truncated for e in G])
        self.final_obs = cat([e.final_obs for e in G])
        self.ep_out = cat([e.ep_out for e in G])
        self.priv = cat([e.priv for e in G])
        self.final_priv = cat([e.final_priv for e in G])

    def set_sliding_grip_low(self, low):
        self.grip_low = low
        for e in self.groups:
            e.set_sliding_grip_low(low)

    def set_pace_low(self, low):
        self.pace_low = low
        for e in self.groups:
            e.set_pace_low(low)

    def set_chase_gap_low(self, low):
        for e in self.groups:
            e.set_chase_gap_low(low)

    def reset(self):
        for e in self.groups:
            e.reset()
        self._gather()
        return self.obs

    def rotate(self):
        """Deals the pool groups new worlds and starts their cars afresh. Returns (observations,
        mask of the cars that were restarted)."""
        mask = self.torch.zeros(self.n, dtype=self.torch.bool)
        if self.pool:
            for g in range(len(self.fixed), self.n_groups):
                self._deal(g, self.pool[self.rng.integers(len(self.pool))])
                self.groups[g].reset()
                mask[g * self.group:(g + 1) * self.group] = True
        self._gather()
        return self.obs, mask

    def step(self, actions):
        for g, e in enumerate(self.groups):
            e.step(actions[g * self.group:(g + 1) * self.group])
        self._gather()
        return self.obs, self.rewards, self.terminated, self.truncated


def make_env(world, sim, n_envs, env_config=None, reward_config=None, seed=0):
    if sim == 'gpu':
        return DriftGpuEnv(world, n_envs, env_config, reward_config, seed)
    return TorchCpuEnv(world, n_envs, env_config, reward_config, seed)
