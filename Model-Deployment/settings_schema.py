"""Monitoring settings: defaults, validation and persistence.

The detection loop reads these values on every frame, so a malformed value
does not fail one request -- it crashes the stream. Everything that reaches
`current_settings` therefore passes through `validate_settings` first.

Standard library only, so the test suite can import it without a model.
"""

import copy
import json
import os

PPE_KEYS = ('helmet', 'vest', 'mask')
COMPLIANCE_MODES = ('strict', 'explicit')

DEFAULT_SETTINGS = {
    'required_ppe': {
        'helmet': True,
        'vest': True,
        'mask': False,
    },
    'non_compliance_delay': 3,
    'instance_reset_timeout': 5,
    # 'strict' flags a required item with no positive detection; 'explicit'
    # waits for an actual NO-* detection before calling it a violation.
    'compliance_mode': 'strict',
    'confidence_threshold': 0.35,
    'enable_tracking': True,
}

# (min, max) for each numeric setting.
NUMERIC_RANGES = {
    'non_compliance_delay': (0, 60),
    'instance_reset_timeout': (1, 600),
    'confidence_threshold': (0.05, 0.95),
}


class SettingsError(ValueError):
    """A settings payload that must not reach the detection loop."""


def _number(payload, key):
    value = payload[key]
    # bool is an int subclass; `true` is not a valid delay.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"{key} must be a number")
    lo, hi = NUMERIC_RANGES[key]
    if not lo <= value <= hi:
        raise SettingsError(f"{key} must be between {lo} and {hi}")
    return value


def validate_settings(payload, base=None):
    """Return a complete, type-checked settings dict.

    Keys absent from `payload` keep their value from `base` (the defaults when
    `base` is None); unknown keys are rejected rather than silently stored.
    """
    if not isinstance(payload, dict):
        raise SettingsError('Settings must be a JSON object')

    unknown = set(payload) - set(DEFAULT_SETTINGS)
    if unknown:
        raise SettingsError(f"Unknown setting(s): {', '.join(sorted(unknown))}")

    result = copy.deepcopy(base if base is not None else DEFAULT_SETTINGS)

    if 'required_ppe' in payload:
        required = payload['required_ppe']
        if (not isinstance(required, dict)
                or set(required) - set(PPE_KEYS)
                or not all(isinstance(v, bool) for v in required.values())):
            raise SettingsError('required_ppe must map helmet/vest/mask to true or false')
        result['required_ppe'] = {**result['required_ppe'], **required}

    for key in NUMERIC_RANGES:
        if key in payload:
            result[key] = _number(payload, key)

    if 'compliance_mode' in payload:
        if payload['compliance_mode'] not in COMPLIANCE_MODES:
            raise SettingsError(f"compliance_mode must be one of {', '.join(COMPLIANCE_MODES)}")
        result['compliance_mode'] = payload['compliance_mode']

    if 'enable_tracking' in payload:
        if not isinstance(payload['enable_tracking'], bool):
            raise SettingsError('enable_tracking must be true or false')
        result['enable_tracking'] = payload['enable_tracking']

    return result


def load_settings(path):
    """Settings from `path`, or the defaults if the file is missing or invalid."""
    if not os.path.exists(path):
        return copy.deepcopy(DEFAULT_SETTINGS)

    try:
        with open(path) as f:
            stored = json.load(f)
        if isinstance(stored, dict):
            # Drop keys an older or newer build wrote, keep everything valid.
            stored = {k: v for k, v in stored.items() if k in DEFAULT_SETTINGS}
        return validate_settings(stored)
    except (OSError, ValueError) as e:
        print(f"[settings] Ignoring {path}: {e}. Using defaults.")
        return copy.deepcopy(DEFAULT_SETTINGS)


def save_settings(path, settings):
    """Write settings atomically, so a crash mid-write never leaves half a file."""
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, 'w') as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp_path, path)
        return True
    except OSError as e:
        print(f"[settings] Error saving settings: {e}")
        return False
