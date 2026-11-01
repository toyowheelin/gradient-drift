"""The simulator still moves the car the way it did when tests/golden/sim_traces.json was written
(sim_traces.py says how to write it again after a physics change that is meant)."""
import json

import numpy as np
import pytest

import sim_traces as st

# How far a path may move before it counts as a physics change. Loose enough for rounding that
# differs between machines (the paths include wall contact and spins, which magnify it), far tighter
# than any real change to the model.
TOLERANCE = {'x': 0.05, 'z': 0.05, 'heading': 0.01, 'yaw': 0.02, 'latv': 0.05, 'speed': 0.05}


@pytest.fixture(scope='module')
def traces(tracks_dir):
    return st.run(tracks_dir)


@pytest.fixture(scope='module')
def golden():
    with open(st.GOLDEN) as f:
        return json.load(f)


def test_same_scenarios(traces, golden):
    assert sorted(golden) == sorted(traces), 'variants changed: write the traces again (python tests/sim_traces.py --write)'
    for v in golden:
        assert sorted(golden[v]) == sorted(traces[v])


@pytest.mark.parametrize('variant', list(st.VARIANTS))
@pytest.mark.parametrize('scenario', list(st.SCENARIOS))
def test_trace_matches(traces, golden, variant, scenario):
    now, then = traces[variant][scenario], golden[variant][scenario]
    problems = []
    for field, tol in TOLERANCE.items():
        a, b = np.asarray(now[field]), np.asarray(then[field])
        assert a.shape == b.shape, f'{field}: {len(a)} samples, the golden trace has {len(b)}'
        diff = np.abs(a - b)
        if field == 'heading':
            diff = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
        if diff.max() > tol:
            i = int(diff.argmax())
            problems.append(f'{field} off by {diff[i]:.4f} (> {tol}) at tick {(i + 1) * st.EVERY}: '
                            f'{a[i]:.4f}, was {b[i]:.4f}')
    assert not problems, (f'{variant}/{scenario}: the car moves differently now. If the physics change is meant, '
                          'run python tests/sim_traces.py --write and commit the traces.\n  ' + '\n  '.join(problems))
