"""Frame annotation for the live dashboard.

Ultralytics' built-in `results.plot()` draws every raw box with equal weight,
which is the wrong picture for this system: what an operator needs to see is
which *worker* is out of compliance, not that a "NO-Hardhat" box exists
somewhere on screen. These helpers draw the association result instead -- one
box per person, coloured by that person's own compliance state and labelled
with their track ID and the PPE they are missing.
"""

import cv2

COMPLIANT_COLOR = (0, 200, 0)      # BGR
VIOLATION_COLOR = (0, 0, 255)
PENDING_COLOR = (0, 165, 255)      # violation inside the debounce window
UNASSIGNED_COLOR = (160, 160, 160)  # equipment matched to nobody

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_label(frame, text, origin, color, scale=0.5, thickness=1):
    """Draw text on a filled background so it stays readable over any scene."""
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    y = max(y, th + baseline + 2)

    cv2.rectangle(frame, (x, y - th - baseline - 2), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - baseline), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_person(frame, person):
    """Outline one worker and label their compliance state."""
    x1, y1, x2, y2 = person['bbox']

    if person.get('violation_confirmed'):
        color = VIOLATION_COLOR
    elif person.get('pending_seconds') is not None:
        color = PENDING_COLOR
    else:
        color = COMPLIANT_COLOR

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    track_id = person.get('track_id')
    who = f"ID {track_id}" if track_id is not None else "Person"

    if person.get('violation_confirmed'):
        detail = "NO " + ", ".join(p.upper() for p in person['missing_ppe'])
    elif person.get('pending_seconds') is not None:
        detail = f"checking {person['pending_seconds']:.1f}s"
    elif person['worn']:
        detail = "+".join(person['worn'])
    else:
        detail = "OK"

    _draw_label(frame, f"{who} | {detail}", (x1, y1), color)


def draw_unassigned(frame, item):
    """Mark PPE that could not be attributed to any worker.

    Drawn deliberately: a helmet on a bench looks identical to a helmet on a
    head to a frame-level counter, and showing it greyed out makes the
    association decision visible during a demo.
    """
    x1, y1, x2, y2 = item['bbox']
    cv2.rectangle(frame, (x1, y1), (x2, y2), UNASSIGNED_COLOR, 1)
    _draw_label(frame, f"unworn {item['ppe_type']}", (x1, y1), UNASSIGNED_COLOR, scale=0.4)


def draw_hud(frame, fps, person_count, violation_count, inference_ms=None):
    """Top-left status readout: throughput, headcount, live violation count.

    `fps=None` marks a still image, where only the inference time means anything.
    """
    perf = []
    if fps is not None:
        perf.append(f"FPS {fps:.1f}")
    if inference_ms is not None:
        perf.append(f"{inference_ms:.1f} ms")

    lines = [
        " | ".join(perf),
        f"Workers {person_count} | Violations {violation_count}",
    ]
    lines = [line for line in lines if line]

    y = 10
    for line in lines:
        _draw_label(frame, line, (10, y + 18), (40, 40, 40), scale=0.55)
        y += 26


def draw_alert_border(frame, text="NON-COMPLIANCE DETECTED"):
    """Red vignette + banner used when any worker has a confirmed violation."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), VIOLATION_COLOR, 20)
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

    cv2.putText(frame, text, (50, frame.shape[0] - 30), FONT, 0.9,
                VIOLATION_COLOR, 3, cv2.LINE_AA)
    return frame


def annotate(frame, instance_result, fps=0.0, inference_ms=None):
    """Draw the full compliance overlay onto a copy of `frame`."""
    annotated = frame.copy()

    for item in instance_result.get('unassigned_ppe', []):
        draw_unassigned(annotated, item)

    for person in instance_result.get('persons', []):
        draw_person(annotated, person)

    if instance_result.get('violation_count'):
        annotated = draw_alert_border(annotated)

    draw_hud(annotated, fps,
             instance_result.get('person_count', 0),
             instance_result.get('violation_count', 0),
             inference_ms)

    return annotated
