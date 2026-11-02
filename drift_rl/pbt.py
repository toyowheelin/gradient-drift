"""
pbt.py - population-based training around train_ppo_gpu.py (Jaderberg et al., "Population Based
Training of Neural Networks", 2017, https://arxiv.org/abs/1711.09846).

    python pbt.py --name pop1 --members 4 --rounds 8 --round-ticks 25000000 \
        --explore ent-coef=0.002 --explore target-kl=0.01 --explore min-std=0.1 \
        -- --track driftplayground_2021 --sim cpu --chase 0.5 ... (the trainer's own options)

Every member is an ordinary run (runs/<name>_m<k>/), trained --round-ticks at a time, one member
after another (one GPU: the members take turns rather than share it). After each round the members
are ranked by --score, the last evaluation of the round in their progress.csv. The bottom quarter
copies the network, optimiser and normalisation of a member from the top quarter (exploit) and
then moves each explored setting by x0.8 or x1.25 from that member's (explore); everyone else keeps
going as they are. The learning rate is not explored: the trainer already adapts it to hold
--target-kl, so target-kl is the knob that moves it.

The settings every member starts with are the --explore values, moved at random for every member
but the first. pbt.json in runs/<name>_pbt/ keeps each round's scores, settings and copies; it is
read again on a restart, which continues from the last finished round.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PERTURB = (0.8, 1.25)


def last_score(run_dir, column, since_ticks):
    """The last value of column in run_dir/progress.csv at or after since_ticks (None if there is none)."""
    path = os.path.join(run_dir, 'progress.csv')
    if not os.path.exists(path):
        return None
    best = None
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get(column)
            if v not in (None, '') and float(row['ticks']) >= since_ticks:
                best = float(v)
    return best


def ticks_done(run_dir):
    path = os.path.join(run_dir, 'progress.csv')
    if not os.path.exists(path):
        return 0
    last = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            last = int(float(row['ticks']))
    return last


def main():
    argv = sys.argv[1:]
    passthrough = []
    if '--' in argv:
        k = argv.index('--')
        argv, passthrough = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True)
    ap.add_argument('--members', type=int, default=4)
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--round-ticks', type=int, default=25_000_000)
    ap.add_argument('--explore', action='append', default=[], metavar='OPTION=VALUE',
                    help='a trainer option (without --) the population searches over, and its starting value')
    ap.add_argument('--score', default='eval_return', help='the progress.csv column members are ranked by')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', help='print the trainer commands instead of running them')
    args = ap.parse_args(argv)
    if any(a in passthrough for a in ('--name', '--resume', '--steps', '--seed')):
        sys.exit('pbt.py sets --name, --resume, --steps and --seed itself: leave them out of the trainer options')
    explore = {}
    for item in args.explore:
        k, _, v = item.partition('=')
        explore[k.lstrip('-')] = float(v)
    if not explore:
        sys.exit('give at least one --explore OPTION=VALUE')
    if args.members < 2:
        sys.exit('--members: 2 or more')

    rng = np.random.default_rng(args.seed)
    pbt_dir = os.path.join(_HERE, 'runs', f'{args.name}_pbt')
    os.makedirs(pbt_dir, exist_ok=True)
    state_path = os.path.join(pbt_dir, 'pbt.json')
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"continuing {args.name}: {len(state['rounds'])} rounds done", flush=True)
    else:
        members = []
        for m in range(args.members):
            h = dict(explore) if m == 0 else {k: v * float(rng.choice(PERTURB)) for k, v in explore.items()}
            members.append(h)
        state = {'members': members, 'rounds': []}
    run_dir = lambda m: os.path.join(_HERE, 'runs', f'{args.name}_m{m}')

    for r in range(len(state['rounds']), args.rounds):
        scores = []
        for m, h in enumerate(state['members']):
            start = ticks_done(run_dir(m))
            cmd = [sys.executable, os.path.join(_HERE, 'train_ppo_gpu.py'), '--name', f'{args.name}_m{m}',
                   '--steps', str(args.round_ticks), '--seed', str(args.seed * 100 + m)]
            ck = os.path.join(run_dir(m), 'checkpoint.pt')
            if os.path.exists(ck):
                cmd += ['--resume', ck]
                cmd += [a for a in passthrough]
            else:
                cmd += passthrough
            for k, v in h.items():
                cmd += [f'--{k}', f'{v:.6g}']
            print(f"\n== round {r + 1}/{args.rounds}, member {m}: " + ' '.join(f'{k}={v:.4g}' for k, v in h.items()), flush=True)
            if args.dry_run:
                print(' '.join(cmd))
                scores.append(float(-m))
                continue
            subprocess.run(cmd, check=True, cwd=_HERE)
            s = last_score(run_dir(m), args.score, start + 1)
            scores.append(s if s is not None else -float('inf'))
            print(f"   member {m}: {args.score} {scores[-1]:.2f}", flush=True)
        order = np.argsort(scores)[::-1]
        q = max(1, args.members // 4)
        top, bottom = [int(i) for i in order[:q]], [int(i) for i in order[-q:]]
        copies = []
        for loser in bottom:
            if loser in top:
                continue
            winner = int(rng.choice(top))
            if not args.dry_run:
                for fname in ('checkpoint.pt', 'policy.json'):
                    src = os.path.join(run_dir(winner), fname)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(run_dir(loser), fname))
            state['members'][loser] = {k: v * float(rng.choice(PERTURB)) for k, v in state['members'][winner].items()}
            copies.append((loser, winner))
        state['rounds'].append({'scores': scores, 'copies': copies,
                                'members': [dict(h) for h in state['members']]})
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=1)
        print(f"round {r + 1}: scores " + ', '.join(f'm{m} {s:.1f}' for m, s in enumerate(scores))
              + ("; " + ', '.join(f'm{a} <- m{b}' for a, b in copies) if copies else ''), flush=True)
    best = int(np.argmax(state['rounds'][-1]['scores'])) if state['rounds'] else 0
    print(f"\nbest member after the last round: m{best} ({run_dir(best)}/policy.json)")


if __name__ == '__main__':
    main()
