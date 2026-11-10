"""The zone inputs (drift_env ZONE_OBS_SLOT): a policy is shown which edge has a drift zone at its
point and 15 m and 30 m on, from the course's zones file, and training shows each episode one of the
course's zone layouts (fast_sim.zone_variants) with the zone pay following the same layout."""
import numpy as np

import drift_env as de
import fast_sim as fs


def test_zone_variants_first_is_given():
    n = 300
    zones = np.array([[1, 10, 40], [-1, 100, 160], [1, 280, 20]], dtype=float)
    v = fs.zone_variants(zones, n, 1.5)
    assert v.shape == (n, fs.ZONE_VARIANTS)
    want = np.zeros(n)
    want[10:41] = 1
    want[100:161] = -1
    want[280:] = 1
    want[:21] = 1
    assert np.array_equal(v[:, 0], want)
    # the other layouts move the zones about but keep them where they were, roughly
    for k in range(1, fs.ZONE_VARIANTS):
        assert not np.array_equal(v[:, k], want) or k == 0
        assert set(np.unique(v[:, k])) <= {-1.0, 0.0, 1.0}
        assert (v[:, k] != 0).sum() <= (want != 0).sum() + 3 * 2 * int(round(fs.ZONE_JITTER_M / 1.5)) + 3
    assert np.array_equal(v, fs.zone_variants(zones, n, 1.5))      # the same every time


def test_zone_inputs_follow_the_layout(tracks_dir, course_names):
    world = fs.load_world(tracks_dir, track=course_names[0])
    assert world.has_course and world.co.shape[1] == fs.ZONE_COL + fs.ZONE_VARIANTS
    assert np.array_equal(world.co[:, 7], world.co[:, fs.ZONE_COL])
    cfg = de.EnvConfig(zone_variants=fs.ZONE_VARIANTS, recovery_start_fraction=0.0)
    env = de.DriftVecEnv(world, n_envs=64, env_config=cfg, reward_config=de.RewardConfig(), seed=3)
    obs = env.reset()
    shown = env.counters[:, de.C_ZONE_VARIANT].astype(int)
    assert len(set(shown)) > 3                                           # layouts are drawn per car
    nearest = env.mids[:, de.M_NEAREST].astype(int)
    side = world.co[nearest, fs.ZONE_COL + shown]
    zs = de.ZONE_OBS_SLOT
    assert np.array_equal(obs[:, zs], (side > 0).astype(float))
    assert np.array_equal(obs[:, zs + 1], (side < 0).astype(float))
    # a server (zone_variants 1) always shows the zones as given
    env1 = de.DriftVecEnv(world, n_envs=16, env_config=de.EnvConfig(), reward_config=de.RewardConfig(), seed=3)
    env1.reset()
    assert not env1.counters[:, de.C_ZONE_VARIANT].any()


def test_zone_inputs_mirror():
    perm, sign = de.mirror_obs_map()
    idx = {name: i for i, name in enumerate(de.OBS_NAMES)}
    for d in de.EDGE_AHEAD_M:
        a, b = idx[f'zone_pos_{int(d)}m'], idx[f'zone_neg_{int(d)}m']
        assert perm[a] == b and perm[b] == a and sign[a] == sign[b] == 1.0
    assert de.OBS_SIZE == de.OBS_SIZE_V8 + de.N_ZONE_OBS and de.ZONE_OBS_SLOT == de.OBS_SIZE_V8
