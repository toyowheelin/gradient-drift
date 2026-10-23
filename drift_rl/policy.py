"""
policy.py - the tiny neural network the drift policy is, plus loading/saving it.

The network is two tanh hidden layers and a linear output:

    x      = clip((obs - obs_mean) / obs_std, -10, 10)
    h1     = tanh(W1 @ x  + b1)
    h2     = tanh(W2 @ h1 + b2)
    action = clip(W3 @ h2 + b3, -1, 1)

That is the whole inference path - about 7,500 multiply-adds per tick for the default 64-unit
layers. The saved .json holds exactly those arrays (row-major, W[i] is the weights INTO output
unit i) plus the observation names and the plugin settings, so the C# side needs nothing but
a file read and three loops.
"""
import json

import numpy as np


class MLP:
    """Fully-connected tanh network, numpy only: what the plugin runs, and what the tools here run
    a policy file with. (Training happens in PyTorch, train_ppo_gpu.py.)"""

    def __init__(self, sizes, rng=None, out_gain=1.0):
        rng = rng or np.random.default_rng(0)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            gain = out_gain if i == len(sizes) - 2 else np.sqrt(2.0)
            a = rng.standard_normal((sizes[i + 1], sizes[i]))
            # orthogonal init
            q, r = np.linalg.qr(a.T if a.shape[0] < a.shape[1] else a)
            q = q * np.sign(np.diag(r))
            w = q.T if a.shape[0] < a.shape[1] else q
            self.W.append(gain * w[:sizes[i + 1], :sizes[i]])
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, x):
        h = x
        n = len(self.W)
        for i in range(n):
            h = h @ self.W[i].T + self.b[i]
            if i < n - 1:
                h = np.tanh(h)
        return h


class Policy:
    """Deterministic deployment policy: observation normalisation + MLP + clip."""

    def __init__(self, mlp, obs_mean, obs_std, meta=None):
        self.mlp = mlp
        self.obs_mean = np.asarray(obs_mean, dtype=np.float64)
        self.obs_std = np.asarray(obs_std, dtype=np.float64)
        self.meta = meta or {}

    def __call__(self, obs):
        # a policy from before an input was added reads only the inputs it was trained with
        x = np.clip((np.atleast_2d(obs)[:, :len(self.obs_mean)] - self.obs_mean) / self.obs_std, -10.0, 10.0)
        return np.clip(self.mlp.forward(x), -1.0, 1.0)

    def save_json(self, path):
        doc = dict(self.meta)
        doc.update({
            'format': 'drift-policy-v1',
            'obs_size': int(len(self.obs_mean)),
            'action_size': int(self.mlp.W[-1].shape[0]),
            'obs_clip': 10.0,
            'obs_mean': self.obs_mean.tolist(),
            'obs_std': self.obs_std.tolist(),
            'layers': [{'activation': 'tanh' if i < len(self.mlp.W) - 1 else 'linear',
                        'W': w.tolist(), 'b': b.tolist()}
                       for i, (w, b) in enumerate(zip(self.mlp.W, self.mlp.b))],
        })
        with open(path, 'w') as f:
            json.dump(doc, f)

    @staticmethod
    def load_json(path):
        with open(path) as f:
            doc = json.load(f)
        sizes = [doc['obs_size']] + [len(l['b']) for l in doc['layers']]
        mlp = MLP(sizes)
        mlp.W = [np.array(l['W'], dtype=np.float64) for l in doc['layers']]
        mlp.b = [np.array(l['b'], dtype=np.float64) for l in doc['layers']]
        meta = {k: v for k, v in doc.items() if k not in ('layers', 'obs_mean', 'obs_std')}
        return Policy(mlp, doc['obs_mean'], doc['obs_std'], meta)


def default_policy_path():
    """The policy the plugin is built with: GradientDriftPlugin/GradientDriftPlugin.policy.json, in an
    AssettoServer source tree beside drift_rl/ or (as in the repository) beside it directly."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    places = [os.path.join(here, '..', 'AssettoServer', 'GradientDriftPlugin', 'GradientDriftPlugin.policy.json'),
              os.path.join(here, '..', 'GradientDriftPlugin', 'GradientDriftPlugin.policy.json')]
    return next((p for p in places if os.path.exists(p)), places[-1])


def warn_if_old_simulator(policy, current_revision, path=''):
    """Prints a warning when a policy was trained before the simulator's last transfer-relevant
    fix (fast_sim.SIMULATOR_REVISION). Returns True if it warned."""
    rev = int(policy.meta.get('simulator_revision', 1))
    if rev >= current_revision:
        return False
    print(f"\nNOTE: {path or 'this policy'} was trained in simulator revision {rev}; the simulator is now at "
          f"revision {current_revision}.\n      Its numbers below are how it does in the CORRECTED simulator. "
          "Retrain before putting it on the server\n      (see README, 'First server test').")
    return True

