"""
sim_target.py - lets fast_sim.py and drift_env.py be compiled for the CPU or for an NVIDIA GPU
from ONE copy of the source.

Both files decorate their compiled functions with `@njit`. Imported normally they get numba's
CPU compiler. `load_variant(name, 'cuda')` executes the SAME file a second time as a separate
module in which `njit` is numba's CUDA device-function compiler instead. So there is exactly one
implementation of the simulator's maths, and the GPU runs that same code, not a second copy.
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VARIANTS = {}


def get_njit(module_globals):
    """Returns (target, njit) for the module being executed."""
    target = module_globals.get('_SIM_TARGET', 'cpu')
    if target == 'cuda':
        from numba import cuda

        def njit(*args, **kwargs):          # accepts @njit and @njit(cache=True, parallel=True)
            if args and callable(args[0]):
                return cuda.jit(device=True)(args[0])
            return cuda.jit(device=True)
        return target, njit
    from numba import njit
    return target, njit


def load_variant(name, target):
    """The module `name`.py from this folder, compiled for `target` ('cpu' or 'cuda')."""
    if target == 'cpu':
        return importlib.import_module(name)
    key = (name, target)
    if key not in _VARIANTS:
        mod_name = f'{name}__{target}'
        spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, name + '.py'))
        mod = importlib.util.module_from_spec(spec)
        mod._SIM_TARGET = target
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        _VARIANTS[key] = mod
    return _VARIANTS[key]


def drop_stale_cache(module_file, dependency_files):
    """Deletes numba's cached compilations of `module_file` if any dependency was edited after
    they were written. Without this, editing fast_sim.py would silently keep running the old
    physics through drift_env's cached loops."""
    import glob
    stem = os.path.splitext(os.path.basename(module_file))[0]
    cached = glob.glob(os.path.join(os.path.dirname(os.path.abspath(module_file)), '__pycache__',
                                    stem + '.*.nb[ci]'))
    if not cached:
        return
    newest_dep = max(os.path.getmtime(f) for f in dependency_files)
    if newest_dep > min(os.path.getmtime(f) for f in cached):
        for f in cached:
            try:
                os.remove(f)
            except OSError:
                pass
