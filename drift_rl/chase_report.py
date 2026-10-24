"""
chase_report.py - how policies chase a ghost leader in the simulator, and how they drive alone.

    python chase_report.py policy.json [more.json ...] --track driftplayground_2021 [--episodes 128]

Each policy drives --episodes training-length episodes alone and the same number behind one of the
track's ghosts (<track>_ghosts.npz, record_ghosts.py), at asked gaps spread over --gaps. A policy
without the chase inputs drives blind behind the ghost: the baseline.

  return         mean episode return (solo / chase; the chase return includes the chase pay)
  drift          share of ticks sliding 20 deg or more
  chase          share of chase-episode ticks with the lead still there (not passed or gone)
  gap            mean gap, the lead's tail to the car's nose, m; in band: within 1 m of the asked gap
  slip / line    mean difference to the lead where it passed the same spot: deg, m
  hits           share of chase episodes ended by touching the lead
"""
import argparse
from dataclasses import replace

import numpy as np

import drift_env as de
import fast_sim as fs
import track_files as tf
from policy import Policy


def run(world, pol, cfg, episodes, seed):
    return de.evaluate(world, pol, n_episodes=episodes, seed=seed, env_config=cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policies', nargs='+')
    ap.add_argument('--track', required=True)
    ap.add_argument('--tracks', default=tf.TRACKS_DIR)
    ap.add_argument('--episodes', type=int, default=128)
    ap.add_argument('--gaps', type=float, nargs=2, default=(0.0, 1.0), metavar=('LOW', 'HIGH'))
    args = ap.parse_args()

    w = fs.load_world(args.tracks, track=args.track)
    w.pace_ref = de.course_pace_reference(args.tracks, args.track, w)
    w.ghosts = de.load_ghosts(args.tracks, args.track)
    if w.ghosts is None:
        raise SystemExit(f'no ghosts for {args.track}: record_ghosts.py --track {args.track}')
    print(f"{'policy':<28} {'return':>13} {'drift':>11} {'chase':>6} {'gap':>6} {'in band':>8} "
          f"{'slip':>6} {'line':>6} {'hits':>6}")
    for path in args.policies:
        pol = Policy.load_json(path)
        cfg = replace(de.env_config_for(pol.meta), chase_gap_low=args.gaps[0], chase_gap_high=args.gaps[1])
        solo = run(w, pol, replace(cfg, chase_share=0.0), args.episodes, 7)
        chase = run(w, pol, replace(cfg, chase_share=1.0), args.episodes, 7)
        had = chase['chase_fraction'] > 0
        c = {k: v[had] for k, v in chase.items()}
        weight = c['chase_fraction'] * c['length']
        mean = lambda k: float(np.sum(c[k] * weight) / max(np.sum(weight), 1e-9))
        hits = float(np.mean(c['end_reason'] == 6)) if had.any() else 0.0
        print(f"{path[-28:]:<28} {np.mean(solo['return']):6.1f}/{np.mean(c['return']):6.1f} "
              f"{np.mean(solo['slide_fraction']):5.0%}/{mean('chase_drift_fraction'):4.0%} "
              f"{np.mean(c['chase_fraction']):6.0%} {mean('chase_mean_gap_m'):6.1f} {mean('chase_in_band_fraction'):8.0%} "
              f"{mean('chase_slip_diff_deg'):6.1f} {mean('chase_line_diff_m'):6.2f} {hits:6.1%}")


if __name__ == '__main__':
    main()
