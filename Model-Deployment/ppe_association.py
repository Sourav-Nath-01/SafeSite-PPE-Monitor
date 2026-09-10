"""Geometric association of PPE detections to individual people.

The detector emits independent boxes for people and for PPE items. On its own
that is not enough to answer the question the system actually asks -- "is *this*
worker wearing a helmet?" -- because a helmet lying on a table is, to a
frame-level counter, indistinguishable from a helmet on a head.

This module closes that gap. Every PPE box is assigned to at most one person
box using a containment score modulated by a body-region prior (helmets belong
near the top of a person box, vests across the torso), and PPE that cannot be
assigned to anyone is reported separately instead of silently counting as
compliance.

Pure geometry: numpy/OpenCV/torch are deliberately not imported here so the
logic stays unit-testable without a model or a GPU.
"""

# Canonical PPE types the compliance rules are written against.
HELMET = 'helmet'
VEST = 'vest'
MASK = 'mask'

PPE_TYPES = (HELMET, VEST, MASK)

# Dataset class names -> (canonical type, worn?). The Construction Site Safety
# dataset ships both positive and negative classes, and the negatives are the
# strongest evidence available: "NO-Hardhat" on a person is an explicit
# violation, not merely an absent helmet.
CLASS_MAP = {
    'hardhat': (HELMET, True),
    'hard-hat': (HELMET, True),
    'hard hat': (HELMET, True),
    'helmet': (HELMET, True),
    'no-hardhat': (HELMET, False),
    'no hardhat': (HELMET, False),
    'no-helmet': (HELMET, False),

    'safety vest': (VEST, True),
    'safety-vest': (VEST, True),
    'vest': (VEST, True),
    'no-safety vest': (VEST, False),
    'no-safety-vest': (VEST, False),
    'no-vest': (VEST, False),

    'mask': (MASK, True),
    'face mask': (MASK, True),
    'face-mask': (MASK, True),
    'respirator': (MASK, True),
    'no-mask': (MASK, False),
    'no-face-mask': (MASK, False),
}

PERSON_CLASSES = {'person', 'worker'}

# Vertical band of the person box (as a fraction of its height, measured from
# the top) where each PPE type is expected to sit. A helmet detected across a
# person's knees is almost certainly a different person's helmet, or one on the
# ground.
REGION_PRIORS = {
    HELMET: (0.00, 0.35),
    MASK: (0.00, 0.40),
    VEST: (0.10, 0.80),
}

# Scoring/acceptance knobs. Kept module-level so they can be swept during
# evaluation rather than hunted for inside the functions.
MIN_CONTAINMENT = 0.30      # PPE box must overlap the person box by this much
MIN_ASSOCIATION_SCORE = 0.20  # containment * region prior, after decay
REGION_DECAY = 2.5          # how sharply the region prior falls off outside the band
PERSON_BOX_PAD = 0.08       # helmets often protrude above the person box


def _area(box):
    """Area of an [x1, y1, x2, y2] box; zero for degenerate boxes."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(box_a, box_b):
    """Area of overlap between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(box_a, box_b):
    """Intersection over union of two boxes."""
    inter = _intersection_area(box_a, box_b)
    union = _area(box_a) + _area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def containment(inner, outer):
    """Fraction of `inner` that falls inside `outer`.

    Preferred over IoU here because a helmet is tiny next to a person: their IoU
    is near zero even when the helmet is squarely on the worker's head.
    """
    inner_area = _area(inner)
    if inner_area <= 0:
        return 0.0
    return _intersection_area(inner, outer) / inner_area


def _pad_box(box, pad_fraction):
    """Grow a box by a fraction of its own width/height on every side."""
    x1, y1, x2, y2 = box
    dw = (x2 - x1) * pad_fraction
    dh = (y2 - y1) * pad_fraction
    return [x1 - dw, y1 - dh, x2 + dw, y2 + dh]


def _region_prior(ppe_box, person_box, ppe_type):
    """How well the PPE box's position matches where that PPE should sit.

    Returns 1.0 inside the expected vertical band and decays toward 0 outside
    it, so a badly-placed box can still win if nothing else is competing but
    will lose to a properly-placed candidate.
    """
    band = REGION_PRIORS.get(ppe_type)
    if band is None:
        return 1.0

    px1, py1, px2, py2 = person_box
    person_height = py2 - py1
    if person_height <= 0:
        return 0.0

    ppe_center_y = (ppe_box[1] + ppe_box[3]) / 2.0
    position = (ppe_center_y - py1) / person_height

    lo, hi = band
    if lo <= position <= hi:
        return 1.0

    distance = lo - position if position < lo else position - hi
    return max(0.0, 1.0 - REGION_DECAY * distance)


def association_score(ppe_box, person_box, ppe_type):
    """Confidence that `ppe_box` belongs to `person_box`.

    Zero when the boxes barely overlap, so unworn equipment lying around the
    site never gets attributed to a worker.
    """
    padded_person = _pad_box(person_box, PERSON_BOX_PAD)
    overlap = containment(ppe_box, padded_person)
    if overlap < MIN_CONTAINMENT:
        return 0.0
    return overlap * _region_prior(ppe_box, person_box, ppe_type)


def normalize_class(class_name):
    """Map a raw dataset class name to (canonical PPE type, worn?).

    Returns None for classes that are not PPE (person, vehicle, machinery,
    safety cone).
    """
    return CLASS_MAP.get(class_name.strip().lower())


def is_person(class_name):
    """True if the class name denotes a person."""
    return class_name.strip().lower() in PERSON_CLASSES


class PersonObservation:
    """One person in one frame, with the PPE evidence assigned to them."""

    def __init__(self, bbox, confidence=1.0, track_id=None):
        self.bbox = bbox
        self.confidence = confidence
        self.track_id = track_id
        self.worn = {}              # ppe_type -> confidence of the worn item
        self.explicit_missing = {}  # ppe_type -> confidence of the NO-* detection

    def evaluate(self, required_ppe, strict=True):
        """Return the PPE types this person is missing.

        `required_ppe` is a {ppe_type: bool} mapping. With `strict` on, a
        required item counts as missing when there is no positive detection for
        it; with `strict` off only an explicit NO-* detection counts, which is
        the more forgiving reading for crowded or partially occluded scenes.
        """
        missing = []
        for ppe_type, required in required_ppe.items():
            if not required:
                continue
            if ppe_type in self.worn:
                continue
            if ppe_type in self.explicit_missing or strict:
                missing.append(ppe_type)
        return missing

    def to_dict(self, required_ppe, strict=True):
        """Serializable view for the API, the UI and the violation log."""
        missing = self.evaluate(required_ppe, strict)
        return {
            'track_id': self.track_id,
            'bbox': [int(round(v)) for v in self.bbox],
            'confidence': round(float(self.confidence), 3),
            'worn': sorted(self.worn.keys()),
            'explicit_missing': sorted(self.explicit_missing.keys()),
            'missing_ppe': missing,
            'is_compliant': not missing,
        }


def associate(detections, min_score=MIN_ASSOCIATION_SCORE):
    """Assign every PPE detection to at most one person.

    `detections` is a list of dicts with 'class', 'bbox' and optionally
    'confidence' and 'track_id' -- the shape the Ultralytics loop already
    produces.

    Returns (people, unassigned) where `people` is a list of PersonObservation
    and `unassigned` lists PPE detections that matched nobody (equipment on the
    ground, on a rack, or belonging to a worker the detector missed).

    Assignment is greedy over the global score matrix: the highest-scoring
    (item, person) pair is committed first, so when two workers stand shoulder
    to shoulder each helmet goes to whichever head it actually sits on rather
    than to whichever person happened to be indexed first.
    """
    people = []
    ppe_items = []

    for det in detections:
        class_name = det.get('class', '')
        bbox = det.get('bbox')
        if not bbox:
            continue

        if is_person(class_name):
            people.append(PersonObservation(
                bbox=[float(v) for v in bbox],
                confidence=det.get('confidence', 1.0),
                track_id=det.get('track_id'),
            ))
            continue

        mapped = normalize_class(class_name)
        if mapped is not None:
            ppe_type, worn = mapped
            ppe_items.append({
                'ppe_type': ppe_type,
                'worn': worn,
                'bbox': [float(v) for v in bbox],
                'confidence': det.get('confidence', 1.0),
                'class': class_name,
            })

    if not people:
        return [], ppe_items

    # Score every (item, person) pair once, then commit the best pairs first.
    candidates = []
    for item_idx, item in enumerate(ppe_items):
        for person_idx, person in enumerate(people):
            score = association_score(item['bbox'], person.bbox, item['ppe_type'])
            if score >= min_score:
                candidates.append((score, item_idx, person_idx))

    candidates.sort(key=lambda c: c[0], reverse=True)

    assigned_items = set()
    for score, item_idx, person_idx in candidates:
        if item_idx in assigned_items:
            continue

        item = ppe_items[item_idx]
        person = people[person_idx]
        bucket = person.worn if item['worn'] else person.explicit_missing

        # Two detections of the same type on one person: keep the confident one.
        existing = bucket.get(item['ppe_type'])
        if existing is None or item['confidence'] > existing:
            bucket[item['ppe_type']] = item['confidence']

        assigned_items.add(item_idx)

    # A positive detection outranks a negative one for the same person and type:
    # "Hardhat" and "NO-Hardhat" both firing means the head was seen covered.
    for person in people:
        for ppe_type in list(person.explicit_missing):
            if ppe_type in person.worn:
                del person.explicit_missing[ppe_type]

    unassigned = [ppe_items[i] for i in range(len(ppe_items)) if i not in assigned_items]
    return people, unassigned
