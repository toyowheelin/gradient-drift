"""
scorecard_report.py - the plugin's session scorecards (EnableScorecard) side by side, one row a
session, so server runs read like the simulator's drive_report.py / chase_report.py tables.

    python scorecard_report.py <server>/gradientdrift_scorecards
    python scorecard_report.py a.json b.json --bots          # a row for each bot as well
    python scorecard_report.py <folder> --min-minutes 10     # skip short sessions

Columns are those of the scorecard's .txt (its header explains them): driven minutes, km/h while
moving, slide (slip 20 - 80 deg) and spin (80+) shares of moving time, walls (share of moving time
pushed out of a wall), wall and car hits an hour of driving, chase minutes, chase gap p50 and the
share within 2 m of the asked gap, stock (share of time the server's own AI had the bots), stuck
episodes. A card still being written (the session running) is marked '*'.
"""
import argparse
import glob
import json
import os


def cards(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, '*.json'))) if os.path.isdir(p) else [p]
    out = []
    for f in files:
        with open(f, encoding='utf-8') as fh:
            card = json.load(fh)
        if isinstance(card, dict) and 'all' in card:
            out.append(card)
    return sorted(out, key=lambda c: c['started'])


def pct(v):
    return f"{v:.0%}"


def num(v, fmt):
    return '-' if v is None else format(v, fmt)


def row(label, r):
    hours = r['driven_minutes'] / 60
    car_hits = r['bot_hits'] + r['player_hits']
    return (f"{label[:34]:34s} {r['driven_minutes']:7.0f} {r['kmh']:5.0f} {pct(r['slide']):>6s} {pct(r['spin']):>5s} "
            f"{pct(r['walls']):>6s} {r['wall_hits'] / hours if hours else 0:6.1f} {car_hits / hours if hours else 0:6.1f} "
            f"{r['chase_minutes']:6.0f} {num(r['gap_p50'], '5.1f'):>5s} {pct(r['in_band']):>5s} {pct(r['stock']):>6s} "
            f"{r['stuck']:5d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='scorecard .json files or folders of them')
    ap.add_argument('--bots', action='store_true', help='a row for each bot under each session')
    ap.add_argument('--min-minutes', type=float, default=0.0, help='leave out sessions shorter than this')
    args = ap.parse_args()

    print(f"{'session':34s} {'driven':>7s} {'km/h':>5s} {'slide':>6s} {'spin':>5s} {'walls':>6s} "
          f"{'wall/h':>6s} {'car/h':>6s} {'chase':>6s} {'gap':>5s} {'band':>5s} {'stock':>6s} {'stuck':>5s}")
    for c in cards(args.paths):
        if c['minutes'] < args.min_minutes:
            continue
        mark = '' if c['final'] else '*'
        label = f"{c['started'][5:16]} {c['track']}{mark}"
        print(row(label, c['all']))
        if c.get('note'):
            print(f"  note: {c['note']}")
        if args.bots:
            for b in c['bots']:
                print(row(f"  bot {b['bot']}", b))


if __name__ == '__main__':
    main()
