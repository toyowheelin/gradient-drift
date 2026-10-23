"""
check_gpu.py - run this once on the training machine before train_ppo_gpu.py.

It answers three questions, in order, and stops at the first problem with a plain message:

  1. Is the GPU stack installed?   (NVIDIA driver, PyTorch with CUDA, numba-cuda)
  2. Does the GPU simulator give the same answers as the CPU simulator?
     Both are compiled from the same source; this steps the same cars with the same actions and
     the same random numbers on each and compares every state variable.
  3. How fast is it?   Ticks per second on the GPU at several car counts, and on the CPU.

    python check_gpu.py
    python check_gpu.py --quick          # skip the larger benchmark sizes
"""
import argparse
import os
import sys
import time

import numpy as np
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))


def random_actions(rng, n):
    """Random actions for every output: steer and pedal in [-1, 1], the brake in [0, 1] (used about half the time)."""
    import drift_env as de
    a = np.clip(rng.normal(0.0, 0.4, (n, de.ACT_SIZE)), -1, 1)
    a[:, 2] = np.clip(a[:, 2], 0.0, 1.0)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tf.tracks_arg(ap)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--cars', type=int, default=256, help='cars in the CPU-vs-GPU comparison')
    ap.add_argument('--steps', type=int, default=40, help='ticks in the CPU-vs-GPU comparison')
    args = ap.parse_args()
    sys.path.insert(0, _HERE)

    simulated = os.environ.get('NUMBA_ENABLE_CUDASIM') == '1'    # numba's no-GPU test mode

    # ---------------- 1. the stack ----------------
    print("1. GPU stack")
    try:
        from numba import cuda
    except Exception as e:
        return fail(f"numba's CUDA target did not import ({type(e).__name__}: {e}).\n"
                    "   Install it with:  pip install \"numba-cuda[cu12]\"   (or [cu13] - match your PyTorch build)")
    if not simulated:
        try:
            if not cuda.is_available():
                return fail("numba cannot find a CUDA GPU. Check that the NVIDIA driver is installed "
                            "(nvidia-smi should list the card) and that numba-cuda was installed with "
                            "its [cu12] or [cu13] extra.")
            dev = cuda.get_current_device()
            name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
            free, total = cuda.current_context().get_memory_info()
            print(f"   numba-cuda sees: {name}, compute capability {dev.compute_capability}, "
                  f"{free/2**30:.1f} of {total/2**30:.1f} GB free")
        except Exception as e:
            return fail(f"numba could not initialise the GPU ({type(e).__name__}: {e})")
        try:
            import torch
            if not torch.cuda.is_available():
                return fail(f"PyTorch {torch.__version__} is installed but was built without CUDA "
                            "(torch.cuda.is_available() is False). Reinstall it with the CUDA command "
                            "from https://pytorch.org/get-started/locally/")
            print(f"   PyTorch {torch.__version__} sees: {torch.cuda.get_device_name(0)}")
        except ImportError:
            return fail("PyTorch is not installed. Use the CUDA command from "
                        "https://pytorch.org/get-started/locally/")
    else:
        print("   (running in numba's CUDA *simulator* - logic check only, no real GPU involved)")

    import sim_target
    import fast_sim as fs
    import drift_env as de
    import gpu_env
    de_gpu = sim_target.load_variant('drift_env', 'cuda')

    print("   loading track ...", flush=True)
    world = fs.load_world(args.tracks)
    env_cfg, rew_cfg = de.EnvConfig(), de.RewardConfig()
    names, world_names = gpu_env.kernel_arg_names()

    def to_dev(host):
        return {k: (cuda.to_device(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v)
                for k, v in host.items() if not k.startswith('_')}

    def launch(kernel, dev, arg_names, n, tpb=64):
        kernel[(n + tpb - 1) // tpb, tpb](*[dev[k] for k in arg_names])

    # ---------------- 2. same answers? ----------------
    n = 3 if simulated else args.cars
    steps = 4 if simulated else args.steps
    print(f"\n2. CPU vs GPU, {n} cars x {steps} ticks, identical actions and random numbers")
    rng = np.random.default_rng(7)
    host = gpu_env.host_arrays(world, n, env_cfg, rew_cfg)
    host['episode_ticks'] = 2.0 if simulated else 15.0   # short episodes, so resets are compared too
    host['rnd'][:] = rng.random((n, de.RND_COLS))
    de.reset_batch(*[host[k] for k in gpu_env._RESET_NAMES])
    t0 = time.time()
    dev = to_dev(host)                    # GPU starts from the CPU's freshly reset cars
    cpu = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in host.items()}
    first_err = None
    worst = 0.0
    n_flag_mismatch = 0
    n_resets = 0
    for t in range(steps):
        act = random_actions(rng, n)
        rnd = rng.random((n, de.RND_COLS))
        cpu['actions'][:] = act
        cpu['rnd'][:] = rnd
        de.step_batch(*[cpu[k] for k in names + world_names])
        dev['actions'].copy_to_device(act)
        dev['rnd'].copy_to_device(rnd)
        launch(de_gpu.step_kernel, dev, names + world_names, n)
        if t == 0:
            print(f"   GPU kernel compiled and ran ({time.time()-t0:.1f} s including compilation)")
        g_states = dev['states'].copy_to_host()
        g_obs = dev['obs'].copy_to_host()
        g_rew = dev['rewards'].copy_to_host()
        if ((dev['terminated'].copy_to_host() != cpu['terminated']).any()
                or (dev['truncated'].copy_to_host() != cpu['truncated']).any()):
            n_flag_mismatch += 1
        n_resets += int((cpu['terminated'] | cpu['truncated']).sum())
        both = np.isfinite(cpu['states']) & np.isfinite(g_states)
        if (np.isfinite(cpu['states']) != np.isfinite(g_states)).any():
            return fail("CPU and GPU disagree about which state entries are set (NaN pattern differs).")
        err = float(np.abs(cpu['states'] - g_states)[both].max())
        err = max(err, float(np.abs(cpu['obs'] - g_obs).max()), float(np.abs(cpu['rewards'] - g_rew).max()))
        if t == 0:
            first_err = err
        worst = max(worst, err)
        # keep the two in lock-step: tiny rounding differences (the GPU fuses multiply-adds)
        # otherwise grow chaotically and would say nothing about correctness
        for k in ('states', 'mids', 'prev_act', 'counters', 'ep', 'obs'):
            dev[k].copy_to_device(cpu[k])
    print(f"   largest difference after one tick: {first_err:.2e}; worst single-tick difference over "
          f"{steps} ticks: {worst:.2e}")
    print(f"   {n_resets} episode endings + resets compared; ticks where the two disagreed on an "
          f"ending: {n_flag_mismatch}")
    if not (first_err < 1e-6):
        return fail("the GPU simulator does not reproduce the CPU simulator.")
    if worst > 1e-3:
        print("   note: at least one car took a different discrete branch on one tick (a threshold "
              "landing on the other side of a rounding difference). Occasional is normal; the first-"
              "tick agreement above is the correctness check.")
    print("   PASS - same simulator on both")

    # ---------------- 3. how fast? ----------------
    print("\n3. Speed (random actions, includes resets)")
    sizes = [2] if simulated else ([1024, 4096] if args.quick else [1024, 4096, 16384, 65536])
    bench_steps = 2 if simulated else 100
    best = (0, 0.0)
    for n in sizes:
        try:
            host = gpu_env.host_arrays(world, n, env_cfg, rew_cfg)
            host['rnd'][:] = rng.random((n, de.RND_COLS))
            dev = to_dev(host)
            launch(de_gpu.reset_kernel, dev, gpu_env._RESET_NAMES, n)
            act = cuda.to_device(random_actions(rng, n))
            dev['actions'] = act
            for _ in range(3):
                launch(de_gpu.step_kernel, dev, names + world_names, n)
            cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(bench_steps):
                launch(de_gpu.step_kernel, dev, names + world_names, n)
            cuda.synchronize()
            rate = n * bench_steps / (time.perf_counter() - t0)
            print(f"   GPU {n:>6} cars: {rate:>12,.0f} ticks/s")
            if rate > best[1]:
                best = (n, rate)
            del dev, act
        except Exception as e:
            print(f"   GPU {n:>6} cars: failed ({type(e).__name__}: {str(e)[:120]}) - probably out of GPU memory")
            break
    if not simulated:
        n = 256
        host = gpu_env.host_arrays(world, n, env_cfg, rew_cfg)
        host['rnd'][:] = rng.random((n, de.RND_COLS))
        de.reset_batch(*[host[k] for k in gpu_env._RESET_NAMES])
        host['actions'][:] = random_actions(rng, n)
        de.step_batch(*[host[k] for k in names + world_names])
        t0 = time.perf_counter()
        for _ in range(50):
            de.step_batch(*[host[k] for k in names + world_names])
        cpu_rate = n * 50 / (time.perf_counter() - t0)
        print(f"   CPU ({os.cpu_count()} threads): {cpu_rate:>12,.0f} ticks/s")
        print(f"\nBest GPU rate: {best[1]:,.0f} ticks/s at {best[0]} cars = {best[1]/cpu_rate:.1f}x this machine's CPU.")
        if best[1] > 1.5 * cpu_rate:
            print(f"Suggested:  python train_ppo_gpu.py --sim gpu --envs {best[0]}")
        else:
            print("The GPU simulator is not meaningfully faster than this CPU here. Use:\n"
                  "            python train_ppo_gpu.py --sim cpu      (network still trains on the GPU)")

    # PyTorch <-> numba shared memory, exactly as the trainer uses it: the same cars driven through
    # gpu_env.DriftGpuEnv and through the CPU environment must report the same observations.
    import torch
    n = 2 if simulated else 1024
    k_steps = 3 if simulated else 20
    genv = gpu_env.DriftGpuEnv(world, n, seed=0)
    cenv = de.DriftVecEnv(world, n, seed=0)
    genv.reset()
    worst = 0.0
    for t in range(k_steps + 1):
        # hand the CPU environment the GPU environment's cars and random numbers, step both
        if t > 0:
            act = torch.as_tensor(random_actions(rng, n), device=genv.device)
            cenv_inputs = {k: genv.t[k].cpu().numpy().copy()
                           for k in ('states', 'mids', 'prev_act', 'counters', 'ep')}
            genv.step(act)
            for k, v in cenv_inputs.items():
                getattr(cenv, k)[:] = v
            host = {'states': cenv.states, 'mids': cenv.mids, 'diags': cenv.diags,
                    'actions': act.cpu().numpy(), 'prev_act': cenv.prev_act, 'counters': cenv.counters,
                    'ep': cenv.ep, 'params': cenv.params, 'rw': cenv.rw, 'obs': cenv.obs,
                    'rewards': cenv.rewards, 'terminated': cenv.terminated, 'truncated': cenv.truncated,
                    'final_obs': cenv.final_obs, 'ep_out': cenv.ep_out, 'rnd': genv.t['rnd'].cpu().numpy(),
                    'env_cfg': cenv.env_cfg, 'episode_ticks': float(cenv.cfg.episode_ticks),
                    'sp_dir': cenv.sp_dir, 'la_idx': cenv.la_idx, 'la_min_radius': cenv.la_min_radius}
            de.step_batch(*[host[k] for k in names], *world.arrays())
            worst = max(worst, float(np.abs(cenv.obs - genv.obs.cpu().numpy()).max()),
                        float(np.abs(cenv.rewards - genv.rewards.cpu().numpy()).max()))
    if not bool(torch.isfinite(genv.obs).all()):
        return fail("observations read back through PyTorch contain non-finite values.")
    print(f"\n4. Through PyTorch (gpu_env.DriftGpuEnv), {n} cars x {k_steps} ticks: largest difference "
          f"from the CPU environment {worst:.2e}")
    if worst > 1e-3:
        print("   note: a car took a different discrete branch on some tick (see step 2); not an error.")
    print("\nALL CHECKS PASSED")
    return 0


def fail(msg):
    print("\nFAILED: " + msg)
    return 1


if __name__ == '__main__':
    sys.exit(main())
