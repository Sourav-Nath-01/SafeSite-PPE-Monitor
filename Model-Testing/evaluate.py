"""Held-out test-set evaluation with a per-class breakdown.

The training run reports validation metrics, which the model selection was
tuned against. The number worth quoting is the one from the 82-image test split
that nothing was chosen on -- and the per-class table, because an aggregate
mAP@50 of 0.82 hides a NO-Hardhat recall of 0.64.

Usage:
    python evaluate.py
    python evaluate.py --split val --markdown ../metrics_val.md
"""

import argparse

DEFAULT_WEIGHTS = '../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt'
DEFAULT_DATA = '../Model-Training/Outputs/data.yaml'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS)
    parser.add_argument('--data', default=DEFAULT_DATA)
    parser.add_argument('--split', default='test', choices=['test', 'val', 'train'],
                        help="'test' is the split no model selection touched")
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.001,
                        help='low by design: mAP is computed over the full PR curve')
    parser.add_argument('--iou', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--device', default=None)
    parser.add_argument('--markdown', default='../metrics_test.md',
                        help='where to write the results table')
    return parser.parse_args()


def resolve_data_yaml(data_path):
    """Rewrite the dataset yaml with absolute split paths.

    `Outputs/data.yaml` stores splits relative to itself, which resolves
    differently depending on the working directory Ultralytics is launched
    from. Emitting a resolved copy next to it makes evaluation reproducible
    from anywhere without editing the file the training notebook reads.
    """
    import os
    import yaml

    with open(data_path) as f:
        spec = yaml.safe_load(f)

    base = os.path.dirname(os.path.abspath(data_path))
    root = os.path.abspath(spec.get('path', base)) if spec.get('path') else base

    for split in ('train', 'val', 'test'):
        if split in spec and spec[split]:
            spec[split] = os.path.abspath(os.path.join(root, spec[split]))

    spec.pop('path', None)

    resolved_path = os.path.join(base, 'data.resolved.yaml')
    with open(resolved_path, 'w') as f:
        yaml.safe_dump(spec, f, sort_keys=False)

    return resolved_path


def collect_per_class(metrics, names):
    """Pull per-class P/R/mAP out of the Ultralytics metrics object."""
    rows = []
    # ap_class_index maps result rows back to dataset class ids; classes with no
    # test-set instances are absent from the results entirely.
    for row_index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(row_index)
        rows.append({
            'class': names[int(class_id)],
            'precision': float(precision),
            'recall': float(recall),
            'map50': float(ap50),
            'map50_95': float(ap),
        })

    rows.sort(key=lambda r: r['map50_95'])
    return rows


def render_markdown(args, metrics, rows):
    """Build a README-ready results table, weakest class first."""
    lines = [
        f"### Held-out `{args.split}` split results",
        "",
        f"Weights: `{args.weights}`  ",
        f"Image size: {args.imgsz} | NMS IoU: {args.iou}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {metrics.box.mp:.4f} |",
        f"| Recall | {metrics.box.mr:.4f} |",
        f"| mAP@50 | {metrics.box.map50:.4f} |",
        f"| mAP@50-95 | {metrics.box.map:.4f} |",
        "",
        "#### Per-class (ascending mAP@50-95 - weakest classes first)",
        "",
        "| Class | Precision | Recall | mAP@50 | mAP@50-95 |",
        "|---|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['class']} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['map50']:.3f} | {row['map50_95']:.3f} |"
        )

    if rows:
        worst = rows[0]
        lines += [
            "",
            f"Weakest class: **{worst['class']}** "
            f"(mAP@50-95 {worst['map50_95']:.3f}, recall {worst['recall']:.3f}).",
        ]

    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()

    from ultralytics import YOLO

    data = resolve_data_yaml(args.data)
    print(f"Evaluating on the '{args.split}' split defined by {data}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    names = model.names
    rows = collect_per_class(metrics, names)
    report = render_markdown(args, metrics, rows)

    print(report)

    if args.markdown:
        with open(args.markdown, 'w') as f:
            f.write(report)
        print(f"Wrote {args.markdown}")


if __name__ == '__main__':
    main()
