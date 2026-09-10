"""Tests for person <-> PPE association.

The case that motivates this module is `test_helmet_on_the_ground_*`: a
frame-level counter marks the scene compliant because a helmet was detected
somewhere, while the worker standing next to it is bare-headed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppe_association import (  # noqa: E402
    associate, containment, iou, association_score, normalize_class,
)

REQUIRED = {'helmet': True, 'vest': True, 'mask': False}


def det(class_name, bbox, confidence=0.9, track_id=None):
    """Build a detection dict in the shape the inference loop emits."""
    return {'class': class_name, 'bbox': bbox, 'confidence': confidence, 'track_id': track_id}


# A worker standing at x=100..200, y=100..400. Head occupies roughly y=100..140.
WORKER = det('Person', [100, 100, 200, 400], track_id=1)
HELMET_ON_HEAD = det('Hardhat', [125, 95, 175, 135])
VEST_ON_TORSO = det('Safety Vest', [105, 180, 195, 300])


class TestGeometry(unittest.TestCase):

    def test_containment_of_fully_enclosed_box(self):
        self.assertAlmostEqual(containment([10, 10, 20, 20], [0, 0, 100, 100]), 1.0)

    def test_containment_of_disjoint_box(self):
        self.assertEqual(containment([500, 500, 510, 510], [0, 0, 100, 100]), 0.0)

    def test_containment_is_partial_when_box_straddles_edge(self):
        # Half of the 10x10 box sits outside the container.
        self.assertAlmostEqual(containment([95, 10, 105, 20], [0, 0, 100, 100]), 0.5)

    def test_iou_of_identical_boxes(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_iou_is_near_zero_for_helmet_and_person(self):
        # Why containment is used instead of IoU: a correctly worn helmet still
        # has almost no IoU with the person box.
        self.assertLess(iou(HELMET_ON_HEAD['bbox'], WORKER['bbox']), 0.1)

    def test_degenerate_box_scores_zero(self):
        self.assertEqual(containment([5, 5, 5, 5], [0, 0, 10, 10]), 0.0)


class TestClassMapping(unittest.TestCase):

    def test_positive_and_negative_classes_map_to_same_type(self):
        self.assertEqual(normalize_class('Hardhat'), ('helmet', True))
        self.assertEqual(normalize_class('NO-Hardhat'), ('helmet', False))

    def test_mapping_is_case_and_separator_tolerant(self):
        self.assertEqual(normalize_class('SAFETY VEST'), ('vest', True))
        self.assertEqual(normalize_class('safety-vest'), ('vest', True))

    def test_scene_objects_are_not_ppe(self):
        for class_name in ('vehicle', 'machinery', 'Safety Cone'):
            self.assertIsNone(normalize_class(class_name))


class TestRegionPrior(unittest.TestCase):

    def test_helmet_at_head_height_outscores_helmet_at_knee_height(self):
        head = association_score([125, 95, 175, 135], WORKER['bbox'], 'helmet')
        knee = association_score([125, 330, 175, 370], WORKER['bbox'], 'helmet')
        self.assertGreater(head, knee)

    def test_vest_is_accepted_across_the_torso(self):
        self.assertGreater(
            association_score(VEST_ON_TORSO['bbox'], WORKER['bbox'], 'vest'), 0.5)


class TestAssociation(unittest.TestCase):

    def test_worn_helmet_and_vest_make_a_compliant_worker(self):
        people, unassigned = associate([WORKER, HELMET_ON_HEAD, VEST_ON_TORSO])

        self.assertEqual(len(people), 1)
        self.assertEqual(unassigned, [])
        self.assertEqual(people[0].evaluate(REQUIRED), [])

    def test_helmet_on_the_ground_is_not_credited_to_the_worker(self):
        """The core regression: unworn equipment must not imply compliance."""
        helmet_on_ground = det('Hardhat', [400, 450, 450, 490])
        people, unassigned = associate([WORKER, VEST_ON_TORSO, helmet_on_ground])

        self.assertEqual(people[0].evaluate(REQUIRED), ['helmet'])
        self.assertEqual(len(unassigned), 1)
        self.assertEqual(unassigned[0]['ppe_type'], 'helmet')

    def test_each_worker_is_judged_on_their_own_ppe(self):
        other = det('Person', [300, 100, 400, 400], track_id=2)
        people, _ = associate([WORKER, HELMET_ON_HEAD, VEST_ON_TORSO, other])

        by_track = {p.track_id: p for p in people}
        self.assertEqual(by_track[1].evaluate(REQUIRED), [])
        self.assertEqual(sorted(by_track[2].evaluate(REQUIRED)), ['helmet', 'vest'])

    def test_adjacent_workers_keep_their_own_helmets(self):
        """Greedy global assignment, not first-match, decides shoulder-to-shoulder cases."""
        left = det('Person', [100, 100, 200, 400], track_id=1)
        right = det('Person', [190, 100, 290, 400], track_id=2)
        left_helmet = det('Hardhat', [125, 95, 175, 135])
        right_helmet = det('Hardhat', [215, 95, 265, 135])

        people, unassigned = associate([left, right, left_helmet, right_helmet])

        by_track = {p.track_id: p for p in people}
        self.assertIn('helmet', by_track[1].worn)
        self.assertIn('helmet', by_track[2].worn)
        self.assertEqual(unassigned, [])

    def test_each_ppe_item_is_claimed_by_at_most_one_person(self):
        left = det('Person', [100, 100, 200, 400], track_id=1)
        right = det('Person', [150, 100, 250, 400], track_id=2)
        people, _ = associate([left, right, HELMET_ON_HEAD])

        claimed = [p for p in people if 'helmet' in p.worn]
        self.assertEqual(len(claimed), 1)

    def test_explicit_negative_detection_is_recorded_as_a_violation(self):
        no_helmet = det('NO-Hardhat', [125, 95, 175, 135])
        people, _ = associate([WORKER, no_helmet, VEST_ON_TORSO])

        self.assertIn('helmet', people[0].explicit_missing)
        self.assertEqual(people[0].evaluate(REQUIRED), ['helmet'])

    def test_positive_detection_overrides_a_conflicting_negative(self):
        no_helmet = det('NO-Hardhat', [125, 95, 175, 135], confidence=0.4)
        people, _ = associate([WORKER, HELMET_ON_HEAD, no_helmet, VEST_ON_TORSO])

        self.assertIn('helmet', people[0].worn)
        self.assertNotIn('helmet', people[0].explicit_missing)
        self.assertEqual(people[0].evaluate(REQUIRED), [])

    def test_optional_ppe_is_never_required(self):
        people, _ = associate([WORKER, HELMET_ON_HEAD, VEST_ON_TORSO])
        self.assertNotIn('mask', people[0].evaluate(REQUIRED))

    def test_explicit_mode_ignores_merely_absent_equipment(self):
        people, _ = associate([WORKER, HELMET_ON_HEAD])

        self.assertEqual(people[0].evaluate(REQUIRED, strict=True), ['vest'])
        self.assertEqual(people[0].evaluate(REQUIRED, strict=False), [])

    def test_ppe_with_no_person_in_frame_is_all_unassigned(self):
        people, unassigned = associate([HELMET_ON_HEAD, VEST_ON_TORSO])

        self.assertEqual(people, [])
        self.assertEqual(len(unassigned), 2)

    def test_empty_frame_is_handled(self):
        self.assertEqual(associate([]), ([], []))

    def test_track_id_survives_association(self):
        people, _ = associate([WORKER, HELMET_ON_HEAD])
        self.assertEqual(people[0].track_id, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
