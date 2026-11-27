"""Tests for settings validation.

Settings are read by the detection loop on every frame, so an invalid value
must be rejected at the API instead of crashing the stream later.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings_schema import (DEFAULT_SETTINGS, SettingsError, load_settings,  # noqa: E402
                             save_settings, validate_settings)


class TestValidateSettings(unittest.TestCase):

    def test_empty_payload_returns_the_defaults(self):
        self.assertEqual(validate_settings({}), DEFAULT_SETTINGS)

    def test_partial_payload_keeps_the_other_values(self):
        result = validate_settings({'non_compliance_delay': 5})

        self.assertEqual(result['non_compliance_delay'], 5)
        self.assertEqual(result['compliance_mode'], DEFAULT_SETTINGS['compliance_mode'])

    def test_partial_required_ppe_is_merged(self):
        result = validate_settings({'required_ppe': {'mask': True}})

        self.assertEqual(result['required_ppe'], {'helmet': True, 'vest': True, 'mask': True})

    def test_defaults_are_not_mutated(self):
        validate_settings({'required_ppe': {'helmet': False}})

        self.assertTrue(DEFAULT_SETTINGS['required_ppe']['helmet'])

    def test_full_payload_from_the_settings_page_is_accepted(self):
        payload = {
            'required_ppe': {'helmet': True, 'vest': False, 'mask': True},
            'non_compliance_delay': 2.5,
            'instance_reset_timeout': 10,
            'compliance_mode': 'explicit',
            'confidence_threshold': 0.5,
            'enable_tracking': False,
        }
        self.assertEqual(validate_settings(payload), payload)

    def test_rejects(self):
        bad_payloads = {
            'not an object': ['helmet'],
            'unknown key': {'debug': True},
            'required_ppe not a dict': {'required_ppe': 'garbage'},
            'unknown PPE type': {'required_ppe': {'gloves': True}},
            'non-bool PPE flag': {'required_ppe': {'helmet': 'yes'}},
            'string number': {'non_compliance_delay': '3'},
            'bool as number': {'non_compliance_delay': True},
            'negative delay': {'non_compliance_delay': -1},
            'confidence above 1': {'confidence_threshold': 1.5},
            'unknown mode': {'compliance_mode': 'lenient'},
            'non-bool tracking': {'enable_tracking': 1},
        }
        for label, payload in bad_payloads.items():
            with self.subTest(label):
                with self.assertRaises(SettingsError):
                    validate_settings(payload)


class TestSettingsFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, 'settings.json')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_round_trip(self):
        settings = validate_settings({'confidence_threshold': 0.6})
        self.assertTrue(save_settings(self.path, settings))

        self.assertEqual(load_settings(self.path), settings)

    def test_missing_file_gives_defaults(self):
        self.assertEqual(load_settings(self.path), DEFAULT_SETTINGS)

    def test_corrupt_file_falls_back_to_defaults(self):
        with open(self.path, 'w') as f:
            f.write('{not json')

        self.assertEqual(load_settings(self.path), DEFAULT_SETTINGS)

    def test_invalid_stored_value_falls_back_to_defaults(self):
        # What the unvalidated API could previously write to disk.
        with open(self.path, 'w') as f:
            json.dump({'required_ppe': 'garbage'}, f)

        self.assertEqual(load_settings(self.path), DEFAULT_SETTINGS)

    def test_stale_keys_are_dropped_not_fatal(self):
        with open(self.path, 'w') as f:
            json.dump({'confidence_threshold': 0.4, 'removed_option': 1}, f)

        self.assertEqual(load_settings(self.path)['confidence_threshold'], 0.4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
