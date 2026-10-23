"""Entry point for the packaged setup (PyInstaller cannot start from a package's __main__.py)."""
import os
import sys

if getattr(sys, 'frozen', False):
    # the track tools are compiled by numba on first use; keep that cache where the user can write
    base = os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(base, 'gradient-drift-setup', 'numba'))

from gradientdrift_setup.cli import main

raise SystemExit(main())
