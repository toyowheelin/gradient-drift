"""
chase_report.py - how policies chase a ghost leader in the simulator, and how they drive alone.

    python chase_report.py policy.json [more.json ...] --track driftplayground_2021 [--episodes 128]

Each policy drives --episodes training-length episodes alone and the same number behind one of the
track's ghosts (<track>_ghosts.npz, record_ghosts.py), at asked gaps spread over --gaps. A policy
without the chase inputs drives blind behind the ghost: the baseline.

  return         mean episode return (solo / chase; the chase return includes the chase pay)
  drift          share of ticks sliding 20 deg or more
  chase          share of chase-episode ticks with the lead still there (not passed or gone)
  gap            mean gap between the two cars' outlines, m; in band: within 1 m of the gap paid for
                 (the asked gap, opened through the lead's transitions: drift_env CHASE_BREATHE_M)
  slip / line    mean difference to the lead where it passed the same spot: deg, m (the line from
                 CHASE_INSIDE_M inside the lead's in the turns)
  hits           share of chase episodes in which the car touched the lead. The first touch ends
                 the episode unless --touches-continue (then a touch held CHASE_CONTACT_LOST_S, 1 s,
                 loses the lead and the car drives on alone)
  km/h           closing speed when a touch starts, mean over touches (what an impact scales with)
  taps           share of touches closing under 3.6 km/h (drift_env.TAP_CLOSING_MS)
  rear/door/nose where touches landed: this car's nose on the lead's tail, side on side, this car's
                 nose into the lead's side (drift_env.TOUCH_END_ZONE_M); the rest are other
  held           share of touches held 1 s (the lead is lost)
  crash          share of touches followed within 3 s by a spin-out or the episode ending
                 (drift_env.TOUCH_CRASH_WINDOW_S)

The lead is a replayed ghost: nothing pushes back, so by default a touch changes neither car and
"crash" only shows what the chaser did next. --touch-push gives the chaser its half of an
equal-mass collision at the contact point (drift_env._touch_push), which is what can turn a tap
into a spin. Use both options together to ask whether the touches would cause crashes.
"""
import argparse
from dataclasses import replace

import numpy as np

import drift_env as de
import fast_sim as fs
import track_files as tf
from policy import Policy


def run(world, pol, cfg, episodes, seed, rcfg=None):
    return de.evaluate(world, pol, n_episodes=episodes, seed=seed, env_config=cfg, reward_config=rcfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policies', nargs='+')
    ap.add_argument('--track', required=True)
    ap.add_argument('--tracks', default=tf.TRACKS_DIR)
    ap.add_argument('--episodes', type=int, default=128)
    ap.add_argument('--gaps', type=float, nargs=2, default=(0.0, 1.0), metavar=('LOW', 'HIGH'))
    ap.add_argument('--scenarios', action='store_true',
                    help='start the chase episodes as training does with --chase-scenarios (drift_env CHASE_SCENARIOS)')
    ap.add_argument('--touches-continue', action='store_true',
                    help='a touch does not end the chase episode (RewardConfig chase_contact_ends 0)')
    ap.add_argument('--touch-push', action='store_true',
                    help='a touch pushes the chaser off the lead (EnvConfig chase_touch_push)')
    de.add_chase_controller_args(ap)
    args = ap.parse_args()
    rcfg = de.RewardConfig(chase_contact_ends=0.0) if args.touches_continue else None

    w = fs.load_world(args.tracks, track=args.track)
    w.pace_ref = de.course_pace_reference(args.tracks, args.track, w)
    w.ghosts = de.load_ghosts(args.tracks, args.track, args.lead_speed)
    if w.ghosts is None:
        raise SystemExit(f'no ghosts for {args.track}: record_ghosts.py --track {args.track}')
    print(f"{'policy':<28} {'return':>13} {'drift':>11} {'chase':>6} {'gap':>6} {'in band':>8} "
          f"{'slip':>6} {'line':>6} {'hits':>6} {'km/h':>5} {'taps':>5} {'rear':>5} {'door':>5} {'nose':>5} "
          f"{'held':>5} {'crash':>5}")
    for path in args.policies:
        pol = Policy.load_json(path)
        cfg = replace(de.env_config_for(pol.meta), chase_gap_low=args.gaps[0], chase_gap_high=args.gaps[1],
                      chase_scenarios=args.scenarios, chase_touch_push=args.touch_push,
                      **de.chase_controller_overrides(args))
        solo = run(w, pol, replace(cfg, chase_share=0.0), args.episodes, 7)
        chase = run(w, pol, replace(cfg, chase_share=1.0), args.episodes, 7, rcfg)
        had = chase['chase_fraction'] > 0
        c = {k: v[had] for k, v in chase.items()}
        weight = c['chase_fraction'] * c['length']
        mean = lambda k: float(np.sum(c[k] * weight) / max(np.sum(weight), 1e-9))
        hits = float(np.mean(c['contacts'] > 0)) if had.any() else 0.0
        touches = max(float(np.sum(c['contacts'])), 1.0)
        closing_kmh = 3.6 * float(np.sum(c['contact_closing_ms'] * c['contacts'])) / touches
        share = lambda k: float(np.sum(c[k])) / touches
        print(f"{path[-28:]:<28} {np.mean(solo['return']):6.1f}/{np.mean(c['return']):6.1f} "
              f"{np.mean(solo['slide_fraction']):5.0%}/{mean('chase_drift_fraction'):4.0%} "
              f"{np.mean(c['chase_fraction']):6.0%} {mean('chase_mean_gap_m'):6.1f} {mean('chase_in_band_fraction'):8.0%} "
              f"{mean('chase_slip_diff_deg'):6.1f} {mean('chase_line_diff_m'):6.2f} {hits:6.1%} "
              f"{closing_kmh:5.1f} {share('contact_taps'):5.0%} {share('contact_rear_end'):5.0%} "
              f"{share('contact_door'):5.0%} {share('contact_nose_side'):5.0%} {share('contact_crashes'):5.0%} "
              f"{share('touch_then_crash'):5.0%}")


if __name__ == '__main__':
    main()
