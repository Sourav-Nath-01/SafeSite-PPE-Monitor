"""Per-worker PPE compliance state machine.

Compliance is decided per tracked person, not per frame. Each track carries its
own debounce timer, so a helmet missed for two frames on a turning head does not
raise an alert, and two workers in the same frame are judged independently --
one violation does not mask the other's compliance.

Geometry lives in ppe_association; this module owns time.
"""

import json
import os
import threading
import time
from datetime import datetime

from ppe_association import associate, PPE_TYPES

COUNTER_FILE = 'instance_counter.json'


class SerialCounter:
    """Daily violation serial, shared by every detector in the process.

    Each video source (the site camera, every browser session) runs its own
    InstanceDetector, but their instance IDs land in one table under a UNIQUE
    constraint. The counter they draw from must therefore be a single object
    behind a lock -- one copy per detector would hand out the same ID twice.
    """

    def __init__(self, path=COUNTER_FILE):
        self.path = path
        self._lock = threading.Lock()
        self.date, self.serial = self._load()

    @staticmethod
    def _today():
        return datetime.now().strftime('%m_%d_%Y')

    def _load(self):
        """Today's serial from disk, restarting the numbering on a new day."""
        today = self._today()
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r') as f:
                    data = json.load(f)
                if data.get('date') == today:
                    print(f"[SerialCounter] Loaded serial counter: {data.get('serial', 0)} for {today}")
                    return today, int(data.get('serial', 0))
        except (OSError, ValueError) as e:
            print(f"[SerialCounter] Error loading counter: {e}")

        print(f"[SerialCounter] Starting new serial counter for {today}")
        return today, 0

    def _save(self):
        """Persist the serial so restarts do not reuse instance IDs."""
        try:
            with open(self.path, 'w') as f:
                json.dump({'date': self.date, 'serial': self.serial}, f)
        except OSError as e:
            print(f"[SerialCounter] Error saving counter: {e}")

    def next_id(self):
        """Next violation ID, formatted MM_DD_YYYY_Serial."""
        with self._lock:
            today = self._today()
            if today != self.date:
                self.date, self.serial = today, 0
            self.serial += 1
            self._save()
            return f"{self.date}_{self.serial}"


class TrackState:
    """Violation state for a single tracked person."""

    def __init__(self, key):
        self.key = key
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.pending_since = None    # when the current violation was first seen
        self.instance_id = None      # set once the violation is confirmed
        self.snapshot_counter = 0
        self.last_missing = []

    @property
    def confirmed(self):
        """True once the violation has outlasted the debounce delay."""
        return self.instance_id is not None


class InstanceDetector:
    """Turn per-frame detections into confirmed, per-worker PPE violations."""

    def __init__(self, window_seconds=5, counter=None):
        self.window_seconds = window_seconds          # track eviction timeout
        self.non_compliance_delay = 3                 # debounce before alerting
        # Pass one SerialCounter to every detector in a process; a detector
        # created on its own gets a private one (the single-source default).
        self.counter = counter if counter is not None else SerialCounter()

        self.required_ppe_settings = {
            'helmet': True,
            'vest': True,
            'mask': False,
        }
        # 'strict': a required item missing from a worker is a violation.
        # 'explicit': only an explicit NO-* detection is a violation, which
        # trades recall for precision on crowded or heavily occluded scenes.
        self.compliance_mode = 'strict'

        self.tracks = {}
        self.current_instance_id = None   # most recent confirmed violation

    def _generate_instance_id(self):
        """Next violation ID, formatted MM_DD_YYYY_Serial."""
        return self.counter.next_id()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def update_settings(self, settings):
        """Apply a settings dict from the API or the settings file."""
        if 'required_ppe' in settings:
            self.required_ppe_settings = settings['required_ppe']
        if 'non_compliance_delay' in settings:
            self.non_compliance_delay = settings['non_compliance_delay']
        if 'instance_reset_timeout' in settings:
            self.window_seconds = settings['instance_reset_timeout']
        if 'compliance_mode' in settings:
            self.compliance_mode = settings['compliance_mode']

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _track_key(self, person, index):
        """Stable identity for a person across frames.

        Falls back to the frame-local index when the tracker is unavailable, so
        the pipeline still runs (with weaker debouncing) on a plain detector.
        """
        if person.track_id is not None:
            return f"id_{person.track_id}"
        return f"idx_{index}"

    def _evict_stale_tracks(self, current_time):
        """Drop tracks that have not been seen within the reset timeout."""
        for key in [k for k, t in self.tracks.items()
                    if current_time - t.last_seen > self.window_seconds]:
            track = self.tracks.pop(key)
            if track.confirmed:
                print(f"[InstanceDetector] Closing instance {track.instance_id} "
                      f"- track {key} gone for >{self.window_seconds}s")

    def process_detection(self, all_detections, dev_mode=False, settings=None):
        """Associate PPE to people, then update each worker's violation state.

        Returns a dict describing the frame: the per-person breakdown under
        'persons', the confirmed violations under 'violations', and the
        aggregate flags the dashboard consumes.
        """
        current_time = time.time()

        if settings:
            self.update_settings(settings)

        strict = self.compliance_mode != 'explicit'
        people, unassigned = associate(all_detections)

        self._evict_stale_tracks(current_time)

        if not people:
            return {
                'instance_id': None,
                'has_person': False,
                'is_compliant': True,
                'missing_ppe': [],
                'detected_ppe': [],
                'should_capture': False,
                'person_count': 0,
                'violation_count': 0,
                'persons': [],
                'violations': [],
                'unassigned_ppe': [self._describe_unassigned(u) for u in unassigned],
            }

        person_views = []
        violations = []
        should_capture = False
        detected_ppe = set()

        for index, person in enumerate(people):
            view = person.to_dict(self.required_ppe_settings, strict)
            detected_ppe.update(view['worn'])

            key = self._track_key(person, index)
            track = self.tracks.get(key)
            if track is None:
                track = TrackState(key)
                self.tracks[key] = track
            track.last_seen = current_time

            missing = view['missing_ppe']

            if missing:
                track.last_missing = missing
                if track.pending_since is None:
                    track.pending_since = current_time

                elapsed = current_time - track.pending_since
                if elapsed >= self.non_compliance_delay:
                    if not track.confirmed:
                        track.instance_id = self._generate_instance_id()
                        track.snapshot_counter = 0
                        print(f"[InstanceDetector] Violation confirmed for track {key}: "
                              f"{track.instance_id} missing {missing}")
                    view['instance_id'] = track.instance_id
                    view['violation_confirmed'] = True
                    should_capture = True
                    violations.append({
                        'instance_id': track.instance_id,
                        'track_key': key,
                        'track_id': view['track_id'],
                        'missing_ppe': missing,
                        'worn': view['worn'],
                        'bbox': view['bbox'],
                    })
                else:
                    # Still inside the debounce window: treat as compliant so a
                    # momentary occlusion never reaches the alert log.
                    view['violation_confirmed'] = False
                    view['pending_seconds'] = round(elapsed, 2)
                    view['is_compliant'] = True
            else:
                track.pending_since = None
                track.instance_id = None
                track.last_missing = []
                view['violation_confirmed'] = False

            person_views.append(view)

        self.current_instance_id = violations[0]['instance_id'] if violations else None

        # Aggregate view, kept in the shape the existing dashboard expects.
        aggregate_missing = sorted({m for v in violations for m in v['missing_ppe']})

        return {
            'instance_id': self.current_instance_id,
            'has_person': True,
            'is_compliant': not violations,
            'missing_ppe': aggregate_missing,
            'detected_ppe': sorted(detected_ppe),
            'should_capture': should_capture,
            'person_count': len(people),
            'violation_count': len(violations),
            'persons': person_views,
            'violations': violations,
            'unassigned_ppe': [self._describe_unassigned(u) for u in unassigned],
        }

    def _describe_unassigned(self, item):
        return describe_unassigned(item)

    def get_next_snapshot_filename(self, instance_id=None):
        """Next snapshot name for a violation, as <instance_id>_snapshot_<n>."""
        if instance_id is None:
            instance_id = self.current_instance_id
        if not instance_id:
            return None

        for track in self.tracks.values():
            if track.instance_id == instance_id:
                track.snapshot_counter += 1
                return f"{instance_id}_snapshot_{track.snapshot_counter}"

        return f"{instance_id}_snapshot_1"


def describe_unassigned(item):
    """PPE that belongs to nobody -- surfaced instead of counted as worn."""
    return {
        'ppe_type': item['ppe_type'],
        'class': item['class'],
        'bbox': [int(round(v)) for v in item['bbox']],
        'confidence': round(float(item['confidence']), 3),
    }


def evaluate_still(detections, required_ppe, strict=True):
    """Compliance for a single image: association, but no tracking or debounce.

    A photo has no "next frame" for a violation to persist into, so every
    worker missing required PPE is reported at once. Nothing is logged and no
    instance ID is drawn -- an uploaded image is a query, not an event.

    Returns the same shape as InstanceDetector.process_detection, so the
    dashboard and the overlay render both without special cases.
    """
    people, unassigned = associate(detections)

    persons = []
    for person in people:
        view = person.to_dict(required_ppe, strict)
        view['violation_confirmed'] = not view['is_compliant']
        persons.append(view)

    violating = [p for p in persons if p['violation_confirmed']]

    return {
        'instance_id': None,
        'has_person': bool(persons),
        'is_compliant': not violating,
        'missing_ppe': sorted({m for p in violating for m in p['missing_ppe']}),
        'detected_ppe': sorted({w for p in persons for w in p['worn']}),
        'should_capture': False,
        'person_count': len(persons),
        'violation_count': len(violating),
        'persons': persons,
        'violations': [{
            'instance_id': None,
            'track_id': p['track_id'],
            'missing_ppe': p['missing_ppe'],
            'worn': p['worn'],
            'bbox': p['bbox'],
        } for p in violating],
        'unassigned_ppe': [describe_unassigned(u) for u in unassigned],
    }


class ComplianceChecker:
    """Reduce a frame result to a single compliant/non-compliant flag."""

    def check_compliance(self, instance_result, dev_mode=False):
        """True when nobody in the frame has a confirmed violation."""
        if not instance_result['has_person']:
            return True
        return instance_result['is_compliant']


class SnapshotManager:
    """Write violation snapshots to disk."""

    def __init__(self, snapshot_dir='snapshots'):
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def save_snapshot(self, frame, filename):
        """Save a frame as <snapshot_dir>/<filename>.jpg; return its path."""
        try:
            import cv2

            if not filename:
                return None

            filepath = os.path.join(self.snapshot_dir, f"{filename}.jpg")

            if frame is not None and frame.size > 0:
                success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if success and os.path.exists(filepath):
                    print(f"Snapshot saved: {filepath}")
                    return filepath
                print(f"Failed to save snapshot: {filepath}")

            return None
        except Exception as e:
            print(f"Error saving snapshot: {e}")
            return None
