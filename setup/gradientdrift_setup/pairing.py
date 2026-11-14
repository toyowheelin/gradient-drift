"""Whether a plugin build can run a policy file, from the features file that comes with the plugin
(GradientDriftPlugin.features.json, made from the plugin's source by tools/plugin_features.py).

The plugin itself refuses a policy it cannot run, but only at the server's start, and then the stock
AI drives the bots with no word anywhere but the server log. Setup checks the pair first and refuses
to install it: the same checks DriftPolicy.TryLoad and ApplyPolicySettings make, in the same order.
"""
import json
import os

FEATURES = 'GradientDriftPlugin.features.json'
FORMAT = 'gradient-drift-plugin-features-v1'


def load_features(folder):
    """The features of the plugin build in a folder, or None when it has no features file (a build
    from before the file existed: its pairing cannot be checked)."""
    path = os.path.join(folder, FEATURES) if folder else ''
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get('format') == FORMAT else None


def policy_needs(doc):
    """What a policy file (its parsed JSON) needs from the plugin, in a few plain words each: the
    list setup shows and the release notes print."""
    hist = doc.get('history') if isinstance(doc.get('history'), dict) else None
    count = (int(hist.get('frames', 0)) * len(hist.get('slots') or [])) if hist else 0
    out = [f"{int(doc.get('obs_size', 0)) - count} inputs" + (f" plus {count} history inputs" if count else ''),
           f"{doc.get('action_size')} outputs"]
    if isinstance(doc.get('residual'), dict):
        out.append('a chase add-on network')
    settings = doc.get('plugin_settings') or {}
    if settings:
        out.append('settings ' + ', '.join(f'{k}={v:g}' if isinstance(v, (int, float)) else k for k, v in settings.items()))
    return out


def mismatches(doc, features):
    """Why the plugin with these features cannot run this policy (its parsed JSON): an empty list
    when it can."""
    out = []
    if doc.get('format') != features.get('policy_format'):
        return [f"the policy file's format is '{doc.get('format')}', the plugin reads '{features.get('policy_format')}'"]
    obs = int(doc.get('obs_size', 0) or 0)
    actions = doc.get('action_size')
    hist = doc.get('history')
    frames = 0
    if isinstance(hist, dict):
        can = features.get('history')
        frames = int(hist.get('frames', 0) or 0)
        count = frames * len(hist.get('slots') or [])
        if not can:
            out.append("the policy reads a short history of its inputs; this plugin build cannot feed one")
        elif (obs - count not in can.get('base_obs_sizes', [can['base_obs_size']]) or count > can['max_inputs']
              or frames < 1):
            bases = ' or '.join(map(str, can.get('base_obs_sizes', [can['base_obs_size']])))
            out.append(f"the policy's history block ({count} inputs after {obs - count}) is not one this plugin runs "
                       f"(at most {can['max_inputs']} after {bases})")
        obs -= count
    if obs not in features.get('obs_sizes', []):
        out.append(f"the policy reads {obs} inputs; this plugin build gives {', '.join(map(str, features.get('obs_sizes', [])))}")
    if actions not in features.get('action_sizes', []):
        out.append(f"the policy has {actions} outputs; this plugin build drives with "
                   f"{' or '.join(map(str, features.get('action_sizes', [])))}")
    mode = doc.get('control_mode', 'residual')
    if mode not in features.get('control_modes', []):
        out.append(f"the policy is a '{mode}' policy; this plugin runs {', '.join(features.get('control_modes', []))} ones")
    res = doc.get('residual')
    if isinstance(res, dict):
        can = features.get('residual')
        if not can:
            out.append("the policy has a chase add-on network; this plugin build cannot run one")
        elif (res.get('gate_input') != can['gate_input'] or obs not in can.get('obs_sizes', [can['obs_size']])
              or (frames > 0 and not can.get('with_history'))):
            out.append("the policy's chase add-on network is not one this plugin build runs")
    unknown = [k for k in (doc.get('plugin_settings') or {}) if k not in features.get('plugin_settings', [])]
    if unknown:
        out.append(f"the policy was trained with {', '.join(unknown)}, which this plugin build does not have")
    return out


def check_files(policy_path, plugin_folder):
    """(mismatches, whether it could be checked) for a policy file and the plugin build in a folder."""
    features = load_features(plugin_folder)
    if features is None:
        return [], False
    try:
        with open(policy_path, encoding='utf-8') as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        return [f"the policy file cannot be read ({e})"], True
    return mismatches(doc, features), True
