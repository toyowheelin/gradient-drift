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

--by-lead also splits each policy's chase episodes by who drove the lead (the ghost file's labels:
a bot at a pace and skill, a public driver, the owner's telemetria) and lists the leads with the
most touches beside how each lead drives: GT Sophy and Sony AI found that the opponents a racer
practises against decide both how often it collides and whether it learns to hang back. Use many
episodes (e.g. 2048): each lead kind needs dozens.

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


def lead_traits(g):
    """How a ghost drives, from its own samples: mean km/h, share sliding over 20 deg, worst distance
    from the AI line (m), share braking (brake input over 0.2), steering reversals a minute, and its
    95th-percentile yaw acceleration (deg/s2) - the last two a measure of how jerky it is."""
    t = g[:, de.GH_T]
    ok = np.r_[True, np.diff(t) > 0]
    g, t = g[ok], t[ok]
    minutes = max((t[-1] - t[0]) / 60.0, 1e-9)
    steer = g[:, de.GH_STEER]
    s = np.sign(np.round(steer, 3))
    s = s[s != 0]
    reversals = int(np.sum(s[1:] != s[:-1])) / minutes
    yaw_acc = np.abs(np.diff(g[:, de.GH_YAW]) / np.maximum(np.diff(t), 1e-3))
    return (float(np.mean(g[:, de.GH_SPEED])) * 3.6, float(np.mean(np.abs(g[:, de.GH_SLIP]) > np.radians(20.0))),
            float(np.max(np.abs(g[:, de.GH_CTE]))), float(np.mean(g[:, de.GH_BRAKE] > 0.2)), reversals,
            float(np.degrees(np.quantile(yaw_acc, 0.95))) if len(yaw_acc) else 0.0)


def kind_group(label):
    """'bot pace 0.90 skill 0.5' -> 'bot skill 0.5'; 'public x.csv car 4 (m)' -> 'public x.csv'."""
    base = label.replace(' ?', '')
    speed = ''
    if ' x' in base and base.rsplit(' x', 1)[1].replace('.', '').isdigit():
        base, speed = base.rsplit(' x', 1)
        speed = f' x{speed}'
    if base.startswith('bot') and ' skill ' in base:
        return 'bot skill ' + base.split(' skill ')[1] + speed
    if base.startswith('public') and ' car ' in base:
        return base.split(' car ')[0] + speed
    return base + speed


def by_lead(c, kinds, ghosts):
    """c: the chase episodes (evaluate() dict, already only those that had a lead)."""
    g = c['ghost']
    hit = c['contacts'] > 0
    weight = c['chase_fraction'] * c['length']
    groups = np.array([kind_group(kinds[i]) if 0 <= i < len(kinds) else '?' for i in g])
    print(f"  {'lead kind':<36} {'episodes':>8} {'gap':>6} {'in band':>8} {'drift':>6} {'hits':>6} {'km/h':>5}")
    for name in sorted(set(groups)):
        m = groups == name
        w = weight[m]
        wm = lambda k: float(np.sum(c[k][m] * w) / max(np.sum(w), 1e-9))
        touches = max(float(np.sum(c['contacts'][m])), 1.0)
        kmh = 3.6 * float(np.sum(c['contact_closing_ms'][m] * c['contacts'][m])) / touches
        print(f"  {name[:36]:<36} {int(m.sum()):8d} {wm('chase_mean_gap_m'):6.1f} {wm('chase_in_band_fraction'):8.0%} "
              f"{wm('chase_drift_fraction'):6.0%} {float(np.mean(hit[m])):6.1%} {kmh:5.1f}")
    # single leads with the most touches, beside how each drives
    rows = []
    for i in np.unique(g[g >= 0]):
        m = g == i
        rows.append((int(np.sum(hit[m])), int(m.sum()), int(i)))
    rows.sort(key=lambda r: (-r[0], -r[0] / max(r[1], 1)))
    worst = [r for r in rows if r[0] > 0][:12]
    if worst:
        print(f"  leads touched most: {'hit/eps':>8} {'km/h':>5} {'slide':>6} {'off m':>6} {'brake':>6} {'rev/min':>8} "
              f"{'yaw acc p95':>11}  lead")
        for hits, eps, i in worst:
            kmh, slide, off, brake, rev, yaw = lead_traits(ghosts[i])
            print(f"  {'':18} {hits:3d}/{eps:<4d} {kmh:5.1f} {slide:6.0%} {off:6.1f} {brake:6.0%} {rev:8.1f} {yaw:11.0f}  "
                  f"#{i} {kinds[i] if i < len(kinds) else '?'}")
    # all leads: does how a lead drives go with how often it is touched?
    if len(rows) >= 8:
        traits = np.array([lead_traits(ghosts[i]) for _, _, i in rows])
        rate = np.array([h / max(e, 1) for h, e, _ in rows])
        names = ('km/h', 'slide', 'off line', 'brake', 'reversals', 'yaw acc')
        corr = []
        for k, name in enumerate(names):
            x = traits[:, k]
            corr.append(f"{name} {np.corrcoef(x, rate)[0, 1]:+.2f}" if np.std(x) > 0 and np.std(rate) > 0 else f"{name} -")
        print("  hit rate per lead against how it drives (correlation over " f"{len(rows)} leads): " + ", ".join(corr))


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
    ap.add_argument('--by-lead', action='store_true',
                    help='also split the chase episodes by who drove the lead, and list the leads touched most')
    de.add_chase_controller_args(ap)
    args = ap.parse_args()
    rcfg = de.RewardConfig(chase_contact_ends=0.0) if args.touches_continue else None

    w = fs.load_world(args.tracks, track=args.track)
    w.pace_ref = de.course_pace_reference(args.tracks, args.track, w)
    w.ghosts = de.load_ghosts(args.tracks, args.track, args.lead_speed)
    if w.ghosts is None:
        raise SystemExit(f'no ghosts for {args.track}: record_ghosts.py --track {args.track}')
    kinds = de.load_ghost_kinds(args.tracks, args.track, args.lead_speed) if args.by_lead else None
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
        if kinds is not None and had.any():
            by_lead(c, kinds, w.ghosts)


if __name__ == '__main__':
    main()
