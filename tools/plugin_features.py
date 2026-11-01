"""What policy files this build of the plugin can run, written to
GradientDriftPlugin/GradientDriftPlugin.features.json from the plugin's C# source.

The plugin refuses a policy file it cannot run (the stock AI drives, and the server log says why).
Setup reads the features file that comes with the plugin to refuse such a pair before it installs
anything (setup/gradientdrift_setup/pairing.py), and the release workflow checks the shipped policy
against it. The file is made from the source, so it cannot drift from what the plugin really does:

    python tools/plugin_features.py           # write the file (after changing what the plugin reads)
    python tools/plugin_features.py --check   # exit 1 if the file is out of date (CI runs this)

What it reads from the source:
  - the plugin settings a policy file may carry: the names in ApplyPolicySettings' validation switch
    (GradientDriftPlugin.cs), the one place a setting has to be added to be accepted;
  - the input and action counts DriftPolicy.TryLoad accepts (the ObservationSize* / ActionSize*
    constants it compares against), the history block's base size and limit, and the input that
    gates a residual network.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.join(HERE, '..', 'GradientDriftPlugin')
OUT = os.path.join(PLUGIN, 'GradientDriftPlugin.features.json')
FORMAT = 'gradient-drift-plugin-features-v1'


def _read(name):
    with open(os.path.join(PLUGIN, name), encoding='utf-8-sig') as f:
        return f.read()


def _constants(sources):
    """{name and Class.name: value} of every `const int` in the given {class: source}."""
    raw = {}
    for cls, text in sources.items():
        for name, expr in re.findall(r'\bconst int (\w+)\s*=\s*([^;]+);', text):
            raw[(cls, name)] = expr
    values = {}

    def value(cls, name, depth=0):
        if (cls, name) in values:
            return values[(cls, name)]
        if depth > 20:
            raise ValueError(f'constant {cls}.{name} refers to itself')
        expr = re.sub(r'//.*', '', raw[(cls, name)])

        def ref(m):
            owner, ident = m.group(1), m.group(2)
            owner = owner[:-1] if owner else cls
            if (owner, ident) not in raw:
                raise ValueError(f'{cls}.{name} uses {m.group(0)}, which is not an int constant here')
            return str(value(owner, ident, depth + 1))
        code = re.sub(r'(\w+\.)?([A-Za-z_]\w*)', ref, expr)
        if not re.fullmatch(r'[\d\s+\-*()]+', code):
            raise ValueError(f'cannot work out {cls}.{name} = {expr}')
        values[(cls, name)] = int(eval(code))      # digits and + - * ( ) only, checked above
        return values[(cls, name)]

    return {f'{cls}.{name}': value(cls, name) for cls, name in raw}


def features():
    plugin = _read('GradientDriftPlugin.cs')
    policy = _read('DriftPolicy.cs')
    sources = {'DriftPolicy': policy}
    if os.path.exists(os.path.join(PLUGIN, 'ChaseInputs.cs')):
        sources['ChaseInputs'] = _read('ChaseInputs.cs')
    const = _constants(sources)

    block = re.search(r'bool usable = float\.IsFinite\(value\) && name switch\s*\{(.*?)_ => false', plugin, re.S)
    if not block:
        raise ValueError("ApplyPolicySettings' validation switch not found in GradientDriftPlugin.cs")
    settings = sorted(set(re.findall(r'"(\w+)"\s*=>', block.group(1))))

    load = policy[policy.index('public static DriftPolicy? TryLoad'):]
    obs = sorted({const[f'DriftPolicy.{n}'] for n in re.findall(r'obsSize != (ObservationSize\w*)', load)})
    actions = sorted({const[f'DriftPolicy.{n}'] for n in re.findall(r'actionSize != (ActionSize\w*)', load)})
    out = {
        'format': FORMAT,
        'policy_format': re.search(r'format != "([^"]+)"', load).group(1),
        'control_modes': sorted(set(re.findall(r'controlMode != "(\w+)"', load))),
        'obs_sizes': obs,
        'action_sizes': actions,
        'plugin_settings': settings,
        # GradientDriftPlugin.bundle.zip beside the plugin (PluginData.cs)
        'bundle': os.path.exists(os.path.join(PLUGIN, 'PluginData.cs')),
    }
    m = re.search(r'baseSize != (\w+)', load)
    if m and 'DriftPolicy.MaxHistoryInputs' in const:
        out['history'] = {'base_obs_size': const[f'DriftPolicy.{m.group(1)}'],
                          'max_inputs': const['DriftPolicy.MaxHistoryInputs']}
    m = re.search(r'residualGate != (\w+)\.(\w+) \|\| obsSize != (\w+)', load)
    if m:
        out['residual'] = {'gate_input': const[f'{m.group(1)}.{m.group(2)}'],
                           'obs_size': const[f'DriftPolicy.{m.group(3)}'],
                           'with_history': 'historyFrames > 0' not in load[load.index('residualGate !='):][:200]}
    return out


def text():
    return json.dumps(features(), indent=1) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true', help='exit 1 if the file is out of date, writing nothing')
    args = ap.parse_args(argv)
    want = text()
    have = open(OUT, encoding='utf-8').read() if os.path.exists(OUT) else ''
    if args.check:
        if have.replace('\r\n', '\n') != want:
            print(f'{os.path.relpath(OUT)} is out of date with the plugin source: run python tools/plugin_features.py '
                  'and commit it. What the source says:\n' + want)
            return 1
        print(f'{os.path.relpath(OUT)} matches the plugin source')
        return 0
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write(want)
    print(f'wrote {os.path.relpath(OUT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
