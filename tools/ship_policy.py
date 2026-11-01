"""Make a policy file the one every server runs: the policy the next release ships.

    python tools/ship_policy.py path/to/policy.json --release v0.3.0

One policy goes out with the plugin, the same on every server. This puts it in place as
GradientDriftPlugin/GradientDriftPlugin.policy.json, after checking that the plugin in this source
can run it (the features file, tools/plugin_features.py), and adds its checksum to setup's
SHIPPED_POLICIES (setup/gradientdrift_setup/plan.py), so setup later knows it for a stock policy and
replaces it with a newer release's. The release workflow refuses a tag whose policy is not on that
list or does not fit the plugin. Nothing is written if a check fails. Commit the result.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'setup'))
sys.path.insert(0, HERE)

SHIPPED = os.path.join(ROOT, 'GradientDriftPlugin', 'GradientDriftPlugin.policy.json')
PLAN = os.path.join(ROOT, 'setup', 'gradientdrift_setup', 'plan.py')


def sha256_lf(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read().replace(b'\r\n', b'\n')).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('policy', help='the policy file (a trainer run\'s policy.json)')
    ap.add_argument('--release', required=True, help='the tag the release will have, e.g. v0.3.0')
    args = ap.parse_args(argv)

    import plugin_features
    from gradientdrift_setup import pairing
    if plugin_features.main(['--check']) != 0:
        print('not shipped: the features file is out of date; run python tools/plugin_features.py first')
        return 1
    with open(args.policy, encoding='utf-8') as f:
        doc = json.load(f)
    with open(plugin_features.OUT, encoding='utf-8') as f:
        features = json.load(f)
    problems = pairing.mismatches(doc, features)
    if problems:
        print('not shipped: the plugin in this source cannot run that policy:\n  ' + '\n  '.join(problems))
        return 1

    sha = sha256_lf(args.policy)
    with open(PLAN, encoding='utf-8') as f:
        plan = f.read()
    block = re.search(r'SHIPPED_POLICIES = \{\n(.*?)\n\}', plan, re.S)
    if not block:
        print(f'not shipped: SHIPPED_POLICIES not found in {PLAN}')
        return 1
    if sha not in block.group(1):
        line = f"    '{sha}': '{args.release}',"
        plan = plan[:block.end(1)] + '\n' + line + plan[block.end(1):]
        with open(PLAN, 'w', encoding='utf-8', newline='\n') as f:
            f.write(plan)
        print(f'added {sha} ({args.release}) to SHIPPED_POLICIES')
    else:
        print(f'{sha} is in SHIPPED_POLICIES already')
    if os.path.abspath(args.policy) != os.path.abspath(SHIPPED):
        shutil.copyfile(args.policy, SHIPPED)
    print(f"shipped policy: {doc.get('trained_ticks', 0):,} training ticks; needs "
          + '; '.join(pairing.policy_needs(doc)))
    print('commit GradientDriftPlugin/GradientDriftPlugin.policy.json and setup/gradientdrift_setup/plan.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
