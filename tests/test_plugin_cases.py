"""The plugin checker's cases (tests/plugin/make_cases.py) can be made, and the Python side of
them holds together: the C# plugin is run on them in CI's plugin build (build-plugin.yml)."""
import json
import os
import subprocess
import sys

from conftest import ROOT


def test_cases_are_made(tmp_path):
    subprocess.run([sys.executable, os.path.join(ROOT, 'tests', 'plugin', 'make_cases.py'), str(tmp_path)], check=True)
    with open(tmp_path / 'cases.json') as f:
        cases = json.load(f)['policies']
    loads = [c for c in cases if c['expect'] == 'loads']
    assert len(loads) >= 5 and len(cases) - len(loads) >= 10
    for c in loads:
        assert len(c['obs']) == len(c['actions']) > 0
        assert all(len(a) == 3 and -1 <= a[0] <= 1 and 0 <= a[2] <= 1 for a in c['actions'])
        assert all(os.path.exists(tmp_path / c['file']) for c in cases)
