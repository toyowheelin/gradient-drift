"""The trainer and the plugin agree on the settings a policy file can carry ("plugin_settings"): a
policy trained with a setting the plugin cannot apply is refused on the server, and the stock AI
drives instead. (tests/plugin/ runs the plugin's own loader in CI; this is the quick check.)"""
import json
import os
import re

from conftest import ROOT, SHIPPED_POLICY

# Settings the simulator takes that no plugin build applies: a policy trained with one of them
# cannot run on a server. Listed so a new one shows up here instead of on a server.
SIMULATOR_ONLY = {'EnableWeightTransfer', 'LoadTransferTimeConstantS'}


def plugin_accepts():
    with open(os.path.join(ROOT, 'GradientDriftPlugin', 'DriftPolicy.cs'), encoding='utf-8-sig') as f:
        source = f.read()
    body = source[source.index('public static bool CanApplySetting'):]
    body = body[:body.index('_ => false')]
    return set(re.findall(r'"(\w+)" =>', body))


def test_plugin_applies_every_trainer_setting():
    import drift_env as de
    trainer = set(de.PLUGIN_SETTING_PARAMS)
    accepted = plugin_accepts()
    assert accepted, 'could not read the settings DriftPolicy.CanApplySetting takes'
    missing = trainer - accepted - SIMULATOR_ONLY
    assert not missing, (f'the trainer can write {sorted(missing)} into a policy, but the plugin refuses a policy '
                         'carrying them: add them to DriftPolicy.CanApplySetting and ApplyPolicySettings')
    unknown = accepted - trainer
    assert not unknown, f'the plugin applies {sorted(unknown)}, which the simulator does not model (drift_env.PLUGIN_SETTING_PARAMS)'
    stale = SIMULATOR_ONLY & accepted
    assert not stale, f'{sorted(stale)} are applied by the plugin now: take them out of SIMULATOR_ONLY'


def test_shipped_policy_settings_apply():
    with open(SHIPPED_POLICY) as f:
        settings = set((json.load(f).get('plugin_settings') or {}))
    assert settings <= plugin_accepts(), f'the shipped policy carries settings the plugin refuses: {sorted(settings - plugin_accepts())}'


def test_fitted_tire_correction_numbers_match():
    """The plugin's FittedTireCorrection.cs holds the same numbers as fast_sim's correction."""
    import fast_sim as fs
    with open(os.path.join(ROOT, 'GradientDriftPlugin', 'FittedTireCorrection.cs'), encoding='utf-8-sig') as f:
        source = f.read()
    for name, python in (('Front', fs.FITTED_TIRE_CORRECTION_FRONT), ('Rear', fs.FITTED_TIRE_CORRECTION_REAR)):
        body = source[source.index(f'private static readonly float[] {name} ='):]
        body = body[body.index('{') + 1:body.index('}')]
        plugin = [float(x) for x in re.findall(r'(-?\d+(?:\.\d+)?(?:e-?\d+)?)f', body)]
        assert plugin == list(python), f'FittedTireCorrection.cs {name} differs from fast_sim.FITTED_TIRE_CORRECTION_{name.upper()}'
    assert 'Limit = 0.6f' in source and fs.FITTED_TIRE_CORRECTION_LIMIT == 0.6
