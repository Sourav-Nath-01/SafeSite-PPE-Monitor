"""Latency and throughput benchmark for the PPE detector.

Reports what a deployment decision actually turns on: per-frame latency
percentiles and sustained FPS, for each exported format, on the device you
intend to run on. Mean latency alone hides the tail that causes dropped frames,
so p50/p95/p99 are reported separately.

Usage:
    python benchmark.py                          # PyTorch, auto device
    python benchmark.py --formats pt onnx        # compare export formats
    python benchmark.py --device cpu --runs 200
    python benchmark.py --markdown results.md    # paste-ready table
"""

import argparse
import statistics
import time

DEFAULT_WEIGHTS = '../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights'

FORMAT_SUFFIX = {
    'pt': 'best.pt',
    'onnx': 'best.onnx',
    'engine': 'best.engine',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights-dir', default=DEFAULT_WEIGHTS,
                        help='directory holding the exported weight files')
    parser.add_argument('--formats', nargs='+', default=['pt'],
                        choices=sorted(FORMAT_SUFFIX), help='formats to compare')
    parser.add_argument('--source', default='../Model-Training/Dataset/test/images',
                        help='image directory or single image to run against')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default=None, help="'cpu', '0', ... (default: auto)")
    parser.add_argument('--runs', type=int, default=100, help='timed iterations')
    parser.add_argument('--warmup', type=int, default=15,
                        help='untimed iterations before measuring (CUDA needs these)')
    parser.add_argument('--half', action='store_true', help='FP16 inference')
    parser.add_argument('--markdown', default=None, help='write a markdown table here')
    return parser.parse_args()


def load_frames(source, limit=64):
    """Load images once so file I/O is not counted in the inference timings."""
    import os
    import cv2

    if os.path.isfile(source):
        paths = [source]
    else:
        paths = sorted(
            os.path.join(source, f) for f in os.listdir(source)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )[:limit]

    frames = [cv2.imread(p) for p in paths]
    frames = [f for f in frames if f is not None]

    if not frames:
        raise SystemExit(f"No readable images found at {source}")

    print(f"Loaded {len(frames)} frames from {source}")
    return frames


def synchronize(device):
    """Block until queued GPU work finishes, or CUDA timings are meaningless."""
    import torch
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_format(weights_path, frames, args):
    """Time one weight file and return its latency statistics in milliseconds."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    device = args.device or ('0' if _cuda_available() else 'cpu')

    predict_kwargs = dict(imgsz=args.imgsz, device=device, verbose=False)
    if args.half:
        predict_kwargs['half'] = True

    for i in range(args.warmup):
        model.predict(frames[i % len(frames)], **predict_kwargs)
    synchronize(device)

    latencies = []
    start = time.perf_counter()
    for i in range(args.runs):
        frame = frames[i % len(frames)]

        t0 = time.perf_counter()
        model.predict(frame, **predict_kwargs)
        synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1000)
    wall = time.perf_counter() - start

    latencies.sort()
    return {
        'device': device,
        'mean': statistics.mean(latencies),
        'p50': latencies[int(0.50 * len(latencies))],
        'p95': latencies[int(0.95 * len(latencies))],
        'p99': latencies[int(0.99 * len(latencies))],
        'min': latencies[0],
        'max': latencies[-1],
        'fps': args.runs / wall,
    }


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def render_table(rows):
    """Format results as a markdown table for the README."""
    header = ('| Format | Device | Mean (ms) | p50 | p95 | p99 | FPS |\n'
              '|---|---|---|---|---|---|---|\n')
    body = ''.join(
        f"| {fmt} | {r['device']} | {r['mean']:.1f} | {r['p50']:.1f} | "
        f"{r['p95']:.1f} | {r['p99']:.1f} | {r['fps']:.1f} |\n"
        for fmt, r in rows
    )
    return header + body


def main():
    import os

    args = parse_args()
    frames = load_frames(args.source)

    rows = []
    for fmt in args.formats:
        weights_path = os.path.join(args.weights_dir, FORMAT_SUFFIX[fmt])
        if not os.path.exists(weights_path):
            print(f"[skip] {fmt}: {weights_path} not found "
                  f"(export it first: yolo export model=best.pt format={fmt})")
            continue

        print(f"\nBenchmarking {fmt} ({args.runs} runs, {args.warmup} warmup)...")
        result = benchmark_format(weights_path, frames, args)
        rows.append((fmt, result))

        print(f"  mean {result['mean']:.1f} ms | p95 {result['p95']:.1f} ms | "
              f"{result['fps']:.1f} FPS on {result['device']}")

    if not rows:
        raise SystemExit("Nothing benchmarked - no weight files found.")

    table = render_table(rows)
    print('\n' + table)

    if args.markdown:
        with open(args.markdown, 'w') as f:
            f.write(table)
        print(f"Wrote {args.markdown}")


if __name__ == '__main__':
    main()
