"""Tests for the per-worker violation state machine.

Time is injected by monkey-patching `time.time` so debounce behaviour is
verified deterministically rather than by sleeping.
"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detection_logic  # noqa: E402
from detection_logic import InstanceDetector, ComplianceChecker  # noqa: E402

SETTINGS = {
    'required_ppe': {'helmet': True, 'vest': True, 'mask': False},
    'non_compliance_delay': 3,
    'instance_reset_timeout': 5,
    'compliance_mode': 'strict',
}


def det(class_name, bbox, confidence=0.9, track_id=None):
    """Build a detection dict in the shape the inference loop emits."""
    return {'class': class_name, 'bbox': bbox, 'confidence': confidence, 'track_id': track_id}


def worker(track_id, x_offset=0, helmet=True, vest=True):
    """A worker at a given x offset, optionally wearing helmet and/or vest."""
    x = 100 + x_offset
    frame = [det('Person', [x, 100, x + 100, 400], track_id=track_id)]
    if helmet:
        frame.append(det('Hardhat', [x + 25, 95, x + 75, 135]))
    if vest:
        frame.append(det('Safety Vest', [x + 5, 180, x + 95, 300]))
    return frame


class ClockTest(unittest.TestCase):
    """Base class giving each test a controllable clock and a scratch cwd."""

    def setUp(self):
        self.now = 1000.0
        self._real_time = detection_logic.time.time
        detection_logic.time.time = lambda: self.now

        # InstanceDetector persists its serial counter to the working directory.
        self._prev_cwd = os.getcwd()
        self.tmpdir = tempfile.mkdtemp()
        os.chdir(self.tmpdir)

        self.detector = InstanceDetector()
        self.checker = ComplianceChecker()

    def tearDown(self):
        detection_logic.time.time = self._real_time
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def tick(self, seconds):
        self.now += seconds

    def process(self, detections):
        return self.detector.process_detection(detections, settings=SETTINGS)


class TestDebounce(ClockTest):

    def test_violation_is_not_confirmed_before_the_delay_elapses(self):
        result = self.process(worker(1, helmet=False))

        self.assertTrue(result['is_compliant'])
        self.assertEqual(result['violation_count'], 0)
        self.assertEqual(result['persons'][0]['pending_seconds'], 0.0)

    def test_violation_is_confirmed_once_the_delay_elapses(self):
        self.process(worker(1, helmet=False))
        self.tick(3.0)
        result = self.process(worker(1, helmet=False))

        self.assertFalse(result['is_compliant'])
        self.assertEqual(result['violation_count'], 1)
        self.assertEqual(result['violations'][0]['missing_ppe'], ['helmet'])
        self.assertIsNotNone(result['violations'][0]['instance_id'])

    def test_a_brief_miss_never_raises_an_alert(self):
        """A helmet lost for one frame on a turning head must not alert."""
        self.process(worker(1))
        self.tick(0.2)
        self.process(worker(1, helmet=False))   # single dropped frame
        self.tick(0.2)
        result = self.process(worker(1))

        self.assertTrue(result['is_compliant'])
        self.assertEqual(result['violation_count'], 0)

    def test_recovering_clears_the_pending_timer(self):
        self.process(worker(1, helmet=False))
        self.tick(2.0)
        self.process(worker(1))                 # worker puts the helmet on
        self.tick(2.0)
        result = self.process(worker(1, helmet=False))

        # The timer restarted, so 4s total elapsed still yields no violation.
        self.assertTrue(result['is_compliant'])

    def test_one_worker_yields_one_instance_id_across_many_frames(self):
        self.process(worker(1, helmet=False))
        self.tick(3.0)
        first = self.process(worker(1, helmet=False))

        ids = {first['violations'][0]['instance_id']}
        for _ in range(10):
            self.tick(0.2)
            result = self.process(worker(1, helmet=False))
            ids.add(result['violations'][0]['instance_id'])

        self.assertEqual(len(ids), 1, "a single worker must not be logged repeatedly")


class TestPerWorkerIndependence(ClockTest):

    def test_two_workers_are_judged_separately(self):
        frame = worker(1) + worker(2, x_offset=300, helmet=False)
        self.process(frame)
        self.tick(3.0)
        result = self.process(worker(1) + worker(2, x_offset=300, helmet=False))

        self.assertEqual(result['person_count'], 2)
        self.assertEqual(result['violation_count'], 1)
        self.assertEqual(result['violations'][0]['track_id'], 2)

        by_track = {p['track_id']: p for p in result['persons']}
        self.assertTrue(by_track[1]['is_compliant'])
        self.assertFalse(by_track[2]['is_compliant'])

    def test_each_violating_worker_gets_a_distinct_instance_id(self):
        frame = worker(1, helmet=False) + worker(2, x_offset=300, helmet=False)
        self.process(frame)
        self.tick(3.0)
        result = self.process(frame)

        ids = {v['instance_id'] for v in result['violations']}
        self.assertEqual(len(ids), 2)

    def test_a_compliant_worker_does_not_mask_a_violating_one(self):
        """The frame-level failure mode: one helmet making the scene 'compliant'."""
        frame = worker(1) + worker(2, x_offset=300, helmet=False)
        self.process(frame)
        self.tick(3.0)
        result = self.process(frame)

        self.assertFalse(result['is_compliant'])

    def test_timers_are_independent_across_tracks(self):
        self.process(worker(1, helmet=False))
        self.tick(2.0)
        # Worker 2 appears late; only worker 1 should confirm at t+3.
        self.process(worker(1, helmet=False) + worker(2, x_offset=300, helmet=False))
        self.tick(1.5)
        result = self.process(worker(1, helmet=False) + worker(2, x_offset=300, helmet=False))

        self.assertEqual(result['violation_count'], 1)
        self.assertEqual(result['violations'][0]['track_id'], 1)


class TestTrackLifecycle(ClockTest):

    def test_stale_tracks_are_evicted_after_the_reset_timeout(self):
        self.process(worker(1, helmet=False))
        self.tick(3.0)
        self.process(worker(1, helmet=False))
        self.assertEqual(len(self.detector.tracks), 1)

        self.tick(6.0)          # worker leaves the frame
        self.process([])
        self.assertEqual(len(self.detector.tracks), 0)

    def test_a_returning_worker_opens_a_new_instance(self):
        self.process(worker(1, helmet=False))
        self.tick(3.0)
        first_id = self.process(worker(1, helmet=False))['violations'][0]['instance_id']

        self.tick(6.0)
        self.process([])        # track evicted

        self.process(worker(1, helmet=False))
        self.tick(3.0)
        second_id = self.process(worker(1, helmet=False))['violations'][0]['instance_id']

        self.assertNotEqual(first_id, second_id)

    def test_empty_frame_reports_no_person(self):
        result = self.process([])

        self.assertFalse(result['has_person'])
        self.assertTrue(self.checker.check_compliance(result))
        self.assertEqual(result['persons'], [])


class TestUnassignedEquipment(ClockTest):

    def test_unworn_helmet_is_reported_and_does_not_confer_compliance(self):
        frame = worker(1, helmet=False) + [det('Hardhat', [600, 450, 650, 490])]
        self.process(frame)
        self.tick(3.0)
        result = self.process(frame)

        self.assertEqual(result['violation_count'], 1)
        self.assertEqual(len(result['unassigned_ppe']), 1)
        self.assertEqual(result['unassigned_ppe'][0]['ppe_type'], 'helmet')


class TestSettings(ClockTest):

    def test_optional_ppe_is_not_enforced(self):
        settings = dict(SETTINGS, required_ppe={'helmet': True, 'vest': False, 'mask': False})
        frame = worker(1, vest=False)

        self.detector.process_detection(frame, settings=settings)
        self.tick(3.0)
        result = self.detector.process_detection(frame, settings=settings)

        self.assertTrue(result['is_compliant'])

    def test_explicit_mode_requires_a_negative_detection(self):
        settings = dict(SETTINGS, compliance_mode='explicit')
        frame = worker(1, helmet=False)   # helmet simply absent, not NO-Hardhat

        self.detector.process_detection(frame, settings=settings)
        self.tick(3.0)
        result = self.detector.process_detection(frame, settings=settings)

        self.assertTrue(result['is_compliant'])

    def test_explicit_mode_still_catches_a_negative_detection(self):
        settings = dict(SETTINGS, compliance_mode='explicit')
        frame = worker(1, helmet=False) + [det('NO-Hardhat', [125, 95, 175, 135])]

        self.detector.process_detection(frame, settings=settings)
        self.tick(3.0)
        result = self.detector.process_detection(frame, settings=settings)

        self.assertFalse(result['is_compliant'])

    def test_snapshot_filenames_are_sequential_per_instance(self):
        self.process(worker(1, helmet=False))
        self.tick(3.0)
        result = self.process(worker(1, helmet=False))
        instance_id = result['violations'][0]['instance_id']

        names = [self.detector.get_next_snapshot_filename(instance_id) for _ in range(3)]

        self.assertEqual(names, [f"{instance_id}_snapshot_{i}" for i in (1, 2, 3)])


class TestSharedSerialCounter(ClockTest):
    """Several video sources (browser sessions) write into one instances table."""

    def confirm_violation(self, detector, track_id):
        detector.process_detection(worker(track_id, helmet=False), settings=SETTINGS)
        self.tick(3.0)
        result = detector.process_detection(worker(track_id, helmet=False), settings=SETTINGS)
        return result['violations'][0]['instance_id']

    def test_detectors_sharing_a_counter_never_reuse_an_instance_id(self):
        counter = detection_logic.SerialCounter('shared_counter.json')
        first = InstanceDetector(counter=counter)
        second = InstanceDetector(counter=counter)

        # Both sessions see "worker 1" -- ByteTrack IDs are per tracker, so they collide.
        ids = {self.confirm_violation(first, 1), self.confirm_violation(second, 1)}

        self.assertEqual(len(ids), 2)

    def test_counter_survives_a_restart(self):
        first_id = detection_logic.SerialCounter('restart_counter.json').next_id()
        after_restart = detection_logic.SerialCounter('restart_counter.json').next_id()

        self.assertNotEqual(first_id, after_restart)
        self.assertTrue(after_restart.endswith('_2'))


class TestEvaluateStill(unittest.TestCase):
    """Single uploaded images: association without tracking or debounce."""

    REQUIRED = {'helmet': True, 'vest': True, 'mask': False}

    def test_violation_is_reported_immediately(self):
        result = detection_logic.evaluate_still(worker(None, helmet=False), self.REQUIRED)

        self.assertFalse(result['is_compliant'])
        self.assertEqual(result['violation_count'], 1)
        self.assertEqual(result['missing_ppe'], ['helmet'])
        self.assertTrue(result['persons'][0]['violation_confirmed'])

    def test_compliant_image_has_no_violations(self):
        result = detection_logic.evaluate_still(worker(None), self.REQUIRED)

        self.assertTrue(result['is_compliant'])
        self.assertEqual(result['violations'], [])

    def test_no_instance_id_is_drawn_for_a_still(self):
        result = detection_logic.evaluate_still(worker(None, helmet=False), self.REQUIRED)

        self.assertIsNone(result['instance_id'])
        self.assertIsNone(result['violations'][0]['instance_id'])

    def test_unworn_helmet_still_does_not_confer_compliance(self):
        frame = worker(None, helmet=False) + [det('Hardhat', [600, 500, 650, 540])]

        result = detection_logic.evaluate_still(frame, self.REQUIRED)

        self.assertFalse(result['is_compliant'])
        self.assertEqual(len(result['unassigned_ppe']), 1)

    def test_empty_image(self):
        result = detection_logic.evaluate_still([], self.REQUIRED)

        self.assertFalse(result['has_person'])
        self.assertTrue(result['is_compliant'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
