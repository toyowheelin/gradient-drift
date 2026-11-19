"""Curated chase leads (train_ppo_gpu.py --curated-leads, drift_env.curated_lead_weights): which leads
count as hard, the share of draws they get, and that a chase in the simulator draws by those weights."""
import numpy as np

import drift_env as de


def _ghost(off_line=1.0, brake=0.0, flips_per_min=10.0, seconds=60.0, dt=0.05):
    t = np.arange(0.0, seconds, dt)
    g = np.zeros((len(t), de.GH_COLS))
    g[:, de.GH_T] = t
    g[:, de.GH_SPEED] = 15.0
    g[:, de.GH_CTE] = off_line
    g[:, de.GH_BRAKE] = (np.arange(len(t)) < brake * len(t)) * 0.5
    g[:, de.GH_STEER] = np.sign(np.sin(np.pi * t * flips_per_min / 60.0) + 1e-9) * 0.3
    return g


def test_hard_lead_reasons():
    assert de.hard_lead_reasons(_ghost()) == []
    assert de.hard_lead_reasons(_ghost(off_line=9.0))[0].startswith('off line 9.0')
    assert de.hard_lead_reasons(_ghost(brake=0.6))[0].startswith('brake 60')
    assert 'reversals' in de.hard_lead_reasons(_ghost(flips_per_min=50.0))[0]
    # one glitched sample far off the line does not make a lead hard
    g = _ghost()
    g[10, de.GH_CTE] = 30.0
    assert de.hard_lead_reasons(g) == []


def test_curated_lead_weights():
    ghosts = np.stack([_ghost(), _ghost(), _ghost(), _ghost(off_line=9.0)])
    w, hard = de.curated_lead_weights(ghosts, 0.1)
    assert list(hard) == [False, False, False, True]
    assert np.isclose(w.sum(), 1.0) and np.isclose(w[3], 0.1) and np.allclose(w[:3], 0.3)
    w, _ = de.curated_lead_weights(ghosts, 0.0)
    assert w[3] == 0.0
    # a self-play slot is never hard; no hard leads left = every lead alike
    w, hard = de.curated_lead_weights(ghosts, 0.1, keep_easy=[3])
    assert not hard.any() and np.allclose(w, 0.25)
    # limits can be moved
    _, hard = de.curated_lead_weights(ghosts, 0.1, off_line_m=10.0)
    assert not hard.any()


def test_chases_draw_by_the_weights(tracks_dir, course_names):
    """A chase in the simulator takes its lead by world.lead_weights: here only ghost 2 can be drawn."""
    from dataclasses import replace

    import fast_sim as fs
    import record_ghosts
    from conftest import SHIPPED_POLICY
    from policy import Policy

    course = course_names[0]
    world = fs.load_world(tracks_dir, track=course)
    world.pace_ref = de.course_pace_reference(tracks_dir, course, world)
    pol = Policy.load_json(SHIPPED_POLICY)
    world.ghosts = record_ghosts.record(world, pol, 4, 700, 100, 1.0, 1.0, seed=3)
    world.lead_weights = np.array([0.0, 0.0, 1.0, 0.0])
    cfg, rw = de.server_run_configs(pol.meta, ticks=400, pace=1.0, skill=1.0)
    cfg = replace(cfg, chase_share=1.0)
    env = de.DriftVecEnv(world, n_envs=16, env_config=cfg, reward_config=rw, seed=5)
    obs = env.reset()
    drawn = set()
    for _ in range(200):
        obs, *_ = env.step(np.asarray(pol(obs)))
        drawn |= set(int(g) for g in env.chase_ghost if g >= 0)
    assert drawn == {2}
