# SafeSite - Real-Time PPE Compliance Monitoring

[![tests](https://github.com/Sourav-Nath-01/SafeSite-PPE-Monitor/actions/workflows/tests.yml/badge.svg)](https://github.com/Sourav-Nath-01/SafeSite-PPE-Monitor/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Per-worker PPE compliance monitoring for construction sites: YOLOv8s detection,
ByteTrack worker identity, geometric person-to-PPE association, and a violation
audit trail with snapshots.

![Alt text](assets/videoconstruc2.gif)
> 4,764 workers died on the job in 2020 (3.4 per 100,000 full-time equivalent workers). Workers in transportation and material moving occupations and construction and extraction occupations accounted for nearly half of all fatal occupational injuries (47.4 percent), representing 1,282 and 976 workplace deaths, respectively. 
>
> *Occupational Safety and Health Administration (US Department of Labour)*


## Introduction

### Problem Statement
Construction site safety is a critical concern in the industrial sector. According to the **U.S. Occupational Safety and Health Administration (OSHA)**:
- **4,764 workers** died on the job in 2020 (3.4 per 100,000 full-time equivalent workers)
- **Workers in transportation, material moving, and construction** accounted for nearly **47.4% of fatal occupational injuries**
- **Construction and extraction workers** experienced **976 workplace deaths** in a single year

The primary cause of many preventable injuries and fatalities is **inadequate or improper use of Personal Protective Equipment (PPE)**.

### Solution
This project leverages **state-of-the-art deep learning** with **YoloV8** to automatically detect and verify that construction workers are wearing appropriate PPE in real-time. The system can be deployed as:
- **Automated safety monitoring** on construction sites
- **Real-time alerting** when workers lack proper equipment
- **Compliance reporting** and documentation
- **Risk assessment** for different construction zones

### Compliance Pipeline

Detection alone does not answer the question the system exists to ask. YOLO
reports that a helmet *and* a person are present; it does not report that *this
worker* is wearing *that helmet*. A helmet resting on a bench and a helmet on a
head produce identical frame-level counts.

The pipeline therefore runs four stages:

```
frame
  │
  ├─ 1. Detection      YOLOv8s -> person / PPE / NO-PPE boxes
  │
  ├─ 2. Tracking       ByteTrack -> stable worker ID across frames
  │
  ├─ 3. Association    each PPE box assigned to at most one person, by
  │                    containment score x body-region prior; unmatched
  │                    equipment reported as "unworn", never as compliance
  │
  └─ 4. Temporal logic per-track debounce -> a violation must persist N
                       seconds on the same worker before it is logged
```

**Why each stage matters**

| Stage | Failure it prevents |
|---|---|
| Association | A helmet on the ground marking a bare-headed worker compliant |
| Association | One compliant worker masking a violating one in the same frame |
| Region prior | A helmet at knee height being credited to the person above it |
| Greedy global assignment | Two adjacent workers sharing one helmet detection |
| Tracking | The same worker logged as dozens of separate violations |
| Debounce | A single dropped frame on a turning head raising an alert |

Each of these is covered by a named test in `Model-Deployment/tests/`.

**Association scoring.** IoU is the wrong metric here -- a correctly worn helmet
has near-zero IoU with the person box, because the helmet is tiny by comparison.
The score used instead is:

```
score = containment(ppe_box, person_box) x region_prior(ppe_type, position)

containment  = intersection_area / ppe_box_area
region_prior = 1.0 inside the expected vertical band of the person box,
               decaying linearly outside it
```

Expected bands, as a fraction of person-box height from the top: helmet
`0.00-0.35`, mask `0.00-0.40`, vest `0.10-0.80`. Pairs are committed
highest-score-first across the whole frame, so each PPE item belongs to exactly
one worker and adjacent workers keep their own equipment.

**Compliance modes.** `strict` flags any required item with no positive
detection on that worker (higher recall, the default for live monitoring).
`explicit` waits for an actual `NO-Hardhat`/`NO-Safety Vest`/`NO-Mask` detection
(higher precision on crowded or occluded scenes).

### Dataset
This project utilizes the [**Construction Site Safety Image Dataset**](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) provided by Roboflow, a comprehensive collection of annotated construction site images:

- **Total Samples**: 2,801 professionally annotated images
- **Training Set**: 2,605 images for model training
- **Validation Set**: 114 images for performance monitoring
- **Test Set**: 82 images for final evaluation
- **Annotation Format**: YOLO v8 format with normalized bounding box coordinates

### Detectable Classes
The model is trained to detect **10 distinct classes** across construction sites:

| Safety Equipment | Non-Compliance | Scene Objects |
|-----------------|-----------------|---------------|
| **Hardhat** | NO-Hardhat | Person |
| **Mask** | NO-Mask | Vehicle |
| **Safety Vest** | NO-Safety Vest | Machinery |
| **Safety Cone** | - | - |

This multi-class detection enables comprehensive safety monitoring and risk identification.


![](assets/ppe.webp)


## Setup

The model was trained on a local machine with an **NVIDIA RTX 5060 Ti (16GB VRAM)**. We utilized the `ultralytics` library by [**Ultralytics**](https://docs.ultralytics.com) to run YoloV8 custom object detection on the dataset.

### Training Configuration
- **Model Architecture**: YoloV8s (Small variant for balanced performance)
- **Total Epochs**: 200
- **Batch Size**: 16
- **Base Learning Rate**: 1e-3
- **Optimizer**: Auto (Adaptive optimizer selection)
- **Training Hardware**: NVIDIA RTX 5060 Ti with 16GB VRAM
- **Training Duration**: Approximately 1.92 hours for 200 epochs

### Key Technologies & Libraries
- **Ultralytics YoloV8**: State-of-the-art real-time object detection framework
- **PyTorch**: Deep learning framework backend
- **OpenCV (cv2)**: Image processing and annotation visualization
- **Pandas & Numpy**: Data manipulation and numerical operations
- **Matplotlib & Seaborn**: Advanced data visualization and training analysis
- **Flask**: Web deployment framework for the detection application

**Note:** This repository contains all results, visualizations, trained models, inference code, and comprehensive deployment infrastructure for production-ready PPE detection.

## Project Structure & Folders

### **Model-Training/** - Training Infrastructure & Dataset

Complete training pipeline with dataset and model artifacts.

#### **Training Notebook:**
- **`yolov8-finetuning-for-ppe-detection.ipynb`** (54 cells)
  - Comprehensive Jupyter notebook implementing the complete training workflow
  - **Data Loading & Preprocessing**:
    - Loads dataset in YOLO format
    - Validates image and label files
    - Creates data.yaml configuration
  - **Model Architecture Setup**:
    - Initializes YoloV8s pre-trained model
    - Configures transfer learning parameters
    - Sets up training hyperparameters
  - **Training Phase**:
    - 200 epochs with batch size of 16
    - Real-time loss and metric monitoring
    - Early stopping patience of 25 epochs
    - Learning rate scheduling with decay
  - **Validation & Evaluation**:
    - Per-epoch validation on 114 images
    - Confusion matrix generation
    - Per-class performance metrics
  - **Visualization & Analysis**:
    - Training curve plots (loss, mAP progression)
    - Sample batch visualization
    - Prediction examples on validation set
    - Performance comparison graphs
  - **Model Export**:
    - Best model checkpointing
    - Final model weights storage

#### **Dataset Organization:**
- **`Dataset/train/`** - Training Set (2605 images)
  - `images/`: JPG format training images from construction sites
  - `labels/`: Corresponding YOLO annotation files (.txt format)
    - Format: `[class_id] [center_x] [center_y] [width] [height]` (normalized coordinates)
    - Contains bounding box information for all PPE objects
  
- **`Dataset/valid/`** - Validation Set (114 images)
  - `images/`: JPG format validation images
  - `labels/`: YOLO format validation annotations
  - Used for monitoring model performance during training
  - Helps detect overfitting
  
- **`Dataset/test/`** - Test Set (82 images)
  - `images/`: JPG format test images
  - `labels/`: YOLO format test annotations
  - Reserved for final model evaluation
  - Represents unseen data for unbiased performance assessment

#### **Training Outputs:**
- **`Outputs/data.yaml`** - Dataset Configuration
  ```yaml
  path: ./Dataset
  train: ./Dataset/train
  val: ./Dataset/valid
  test: ./Dataset/test
  nc: 10
  names: ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
          'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
  ```

- **`Outputs/runs/detect/yolov8s_ppe_css_200_epochs/`** - Training Results
  - `confusion_matrix.png`: Per-class detection accuracy heatmap
  - `results.png`: Loss curves and metric progression across 200 epochs
  - `train_batch*.jpg`: Sample detections from training batches
  - `val_batch*_pred.jpg`: Model predictions on validation set
  - `val_batch*_labels.jpg`: Ground truth labels for reference
  - `results.csv`: Detailed epoch-by-epoch metrics

- **`Outputs/training_log_df.csv`** - Complete Training Log
  - 200 rows (one per epoch) with 15 metrics columns:
    - `epoch`: Training epoch number (1-200)
    - `time`: Cumulative training time
    - `train/box_loss`: Bounding box regression loss
    - `train/cls_loss`: Classification loss
    - `train/dfl_loss`: Distribution focal loss
    - `metrics/precision(B)`: Precision score
    - `metrics/recall(B)`: Recall score
    - `metrics/mAP50(B)`: mAP at 50% IoU threshold
    - `metrics/mAP50-95(B)`: mAP across IoU 50-95%
    - `val/box_loss`: Validation box loss
    - `val/cls_loss`: Validation classification loss
    - `val/dfl_loss`: Validation DFL loss
    - `lr/pg0/pg1/pg2`: Learning rates for parameter groups
  - Enables analysis of training dynamics and convergence

#### **Pre-trained Model Weights:**
Stock COCO checkpoints (`yolov8s.pt`, `yolo11n.pt`) are **not** committed --
Ultralytics downloads them automatically on first use. `yolov8s.pt` is the
transfer-learning starting point; `yolo11n.pt` is a lighter alternative worth
benchmarking for edge deployment.

### **Model-Deployment/** - Real-Time Monitoring Service

Flask + SocketIO application that turns the detector into a per-worker
compliance monitor with a reviewable violation log.

#### **Core Application Files:**

- **`ppe_association.py`** - Person <-> PPE association (no ML dependencies)
  - `associate()`: assigns every PPE box to at most one person via greedy
    global matching on `containment x region_prior`
  - `containment()` / `iou()`: box geometry primitives
  - `PersonObservation`: one worker in one frame, with the PPE evidence
    attributed to them and their `evaluate()` compliance verdict
  - `CLASS_MAP`: dataset class names -> `(canonical PPE type, worn?)`, so both
    `Hardhat` and `NO-Hardhat` resolve to the `helmet` requirement
  - Deliberately free of torch/OpenCV/numpy imports, which is what makes the
    logic unit-testable without a GPU, a model or a camera

- **`detection_logic.py`** - Temporal state machine
  - `InstanceDetector`: owns one `TrackState` per tracked worker, each with its
    own debounce timer, violation instance ID and snapshot counter
  - Violations are confirmed only after persisting `non_compliance_delay`
    seconds on the *same* track; tracks are evicted after
    `instance_reset_timeout` seconds out of frame
  - Returns a per-worker breakdown (`persons`), the confirmed `violations`, and
    `unassigned_ppe` for equipment that matched nobody
  - `SnapshotManager`: writes annotated violation frames to `snapshots/`

- **`visualization.py`** - Compliance overlay
  - Draws one box per *worker*, coloured by that worker's own state:
    green compliant, amber inside the debounce window, red confirmed violation
  - Greys out unassigned PPE, making the association decision visible in a demo
  - HUD with live FPS, per-frame inference time, headcount and violation count

- **`app.py`** - Flask service
  - `run_inference()`: `model.track(..., tracker='bytetrack.yaml', persist=True)`
    so worker IDs are stable across frames
  - Rolling 30-frame window for FPS and inference latency, streamed to the UI
  - Per-violation snapshot rate limiting, so one worker cannot flood the queue
  - Weight loading tries the 200-epoch checkpoint first and reports the fallback
    loudly instead of silently serving a weaker model
  - `SECRET_KEY` from `PPE_SECRET_KEY`, random per process otherwise
  - Routes: `/`, `/video_feed`, `/start_stream`, `/stop_stream`, `/stats`,
    `/settings`, `/api/settings`, `/review`, `/api/instances`,
    `/api/instance/<id>/snapshots`, `/snapshots/<file>`

- **`database.py`** - SQLite persistence
  - Tables: `instances`, `snapshots`, `alerts`
  - `_migrate()`: adds `instances.track_id` in place on existing databases, so
    the review page can separate two workers logged in the same minute

- **`tests/`** - 40 unit tests, no model or GPU required
  - `test_ppe_association.py` (23): geometry, class mapping, region priors,
    adjacent-worker assignment, the helmet-on-the-ground regression
  - `test_detection_logic.py` (17): debounce, per-track independence, track
    eviction, instance-ID uniqueness, compliance modes
  - Time is injected by monkey-patching `time.time`, so the temporal tests are
    deterministic rather than sleep-based

#### **Templates:**
- **`index.html`** - Live dashboard: per-worker cards showing ID, worn PPE,
  missing PPE, debounce progress and instance ID, plus the FPS/latency readout
- **`review.html`** - Violation history with snapshots
- **`settings.html`** - Required PPE, debounce delay, reset timeout, compliance
  mode, confidence threshold and tracking toggle

### **Model-Testing/** - Evaluation & Benchmarking

- **`evaluate.py`** - Held-out test-set evaluation
  - Runs on the 82-image `test` split that no model selection touched, and
    prints a per-class table sorted weakest-first, because an aggregate mAP of
    0.88 can hide a class that barely works
  - `resolve_data_yaml()` rewrites the dataset yaml with absolute split paths,
    so evaluation is reproducible from any working directory
  - Usage: `python evaluate.py --split test --markdown ../metrics_test.md`

- **`benchmark.py`** - Latency & throughput benchmark
  - Reports p50/p95/p99 latency and sustained FPS per export format; mean
    latency alone hides the tail that causes dropped frames
  - Warmup iterations and `torch.cuda.synchronize()` before each measurement,
    without which CUDA timings measure queue submission rather than compute
  - Usage: `python benchmark.py --formats pt onnx --device 0 --runs 200`

- **`checker.py`** - CUDA availability and device report
- **`test_cam.py`** - Raw detector on a webcam feed (no compliance logic)
- **`test_single.py`** - Single-image inference

### **assets/**
Repository documentation assets:
- Training visualizations, GIFs, and images used in README

### **Root Files**
- **`requirements.txt`**: Runtime dependencies
- **`requirements-long.txt`**: Install order, including the CUDA-matched PyTorch wheel
- **`run_tests.sh`**: Runs the 40-test compliance suite (no model or GPU needed)
- **`Dockerfile`** / **`.dockerignore`**: CPU container for the monitoring service
- **`README.md`**: Project documentation and usage guide
- **`LICENSE`**: Project licensing information



## Repository Structure

```
Construction-Site-Safety-PPE-Detection/
├── Model-Training/
│   ├── Dataset/{train,valid,test}/{images,labels}/
│   ├── Outputs/
│   │   ├── data.yaml
│   │   └── runs/detect/yolov8s_ppe_css_200_epochs/
│   │       └── weights/best.pt        # committed; .onnx/.engine are exports
│   │       ├── confusion_matrix.png, results.png, Box*_curve.png
│   │       └── results.csv
│   └── yolov8-finetuning-for-ppe-detection.ipynb
│
├── Model-Deployment/
│   ├── ppe_association.py     # person <-> PPE geometry (dependency-free)
│   ├── detection_logic.py     # per-track debounce & violation state machine
│   ├── visualization.py       # compliance overlay + HUD
│   ├── app.py                 # Flask + SocketIO service
│   ├── database.py            # SQLite persistence & migrations
│   ├── templates/{index,review,settings}.html
│   └── tests/
│       ├── test_ppe_association.py
│       └── test_detection_logic.py
│
├── Model-Testing/
│   ├── evaluate.py            # held-out test-split metrics, per class
│   ├── benchmark.py           # p50/p95/p99 latency & FPS per format
│   ├── checker.py, test_cam.py, test_single.py
│
├── Dockerfile
├── run_tests.sh
├── requirements.txt
└── README.md
```

## Training Results & Performance Metrics

### Model Performance (200 Epochs)

The YoloV8s model underwent comprehensive training for **200 epochs** on the construction site safety dataset using an NVIDIA RTX 5060 Ti GPU. The training process completed in approximately **1.92 hours**, demonstrating efficient transfer learning.

#### Final Metrics Achieved (validation split, 114 images):
| Metric | Value |
|--------|-------|
| **Precision** | 0.95 |
| **Recall** | 0.7975 |
| **mAP@50%** | 0.8767 |
| **mAP@50-95%** | 0.6152 |

#### Interpretation:
- **Precision (0.95)**: Of all PPE detections made by the model, 95% are accurate. This indicates the model has excellent reliability in confirming actual PPE when it flags a detection.
- **Recall (0.7975)**: The model identifies approximately 80% of all actual PPE instances in the dataset. This ensures good coverage while maintaining acceptable false negative rates.
- **mAP@50%**: Strong performance at 87.67%, indicating robust detection across all 10 object classes at standard IoU threshold.
- **mAP@50-95%**: Solid performance of 61.52% across stricter IoU thresholds, demonstrating good localization accuracy of bounding boxes.

> **These are validation-split numbers**, and the checkpoint was selected against
> that split, so they are optimistic by construction. Generate the honest
> held-out figures with `python Model-Testing/evaluate.py --split test`, which
> also prints the per-class table -- the aggregate hides that the `Mask` /
> `NO-Mask` classes are far weaker than `Hardhat` and `Person`, since masks are
> small, often occluded and under-represented in the training data.

#### Inference performance

Throughput is a deployment decision, not a footnote: a model that cannot keep up
with the camera drops the frames a violation occurs in. Measure it on your own
hardware before quoting it:

```bash
python Model-Testing/benchmark.py --formats pt onnx --runs 200
```

The benchmark reports p50/p95/p99 latency and sustained FPS per export format.
The live dashboard shows the same two numbers continuously while streaming.

### Training Characteristics:
- **Data Split**: Train: 2605 images | Valid: 114 images | Test: 82 images
- **Batch Processing**: 16 images per batch
- **Hardware Utilization**: NVIDIA RTX 5060 Ti efficiently processed batches with 16GB VRAM
- **Convergence**: Model achieved plateau around epoch 150-180, with final epochs showing stability
- **Classes Detected**: 10 PPE and non-PPE classes including Hardhat, Mask, Safety Vest, Person, Vehicle, Machinery, etc.

### Visualization Results

#### Confusion Matrix
Shows the model's classification accuracy across all 10 PPE classes:

![Alt text](Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/confusion_matrix.png)

#### Training Batch Samples
Visual examples of detected PPE during training batches:

![Alt text](Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/train_batch30972.jpg)

#### Validation Batch Predictions
Model predictions on validation set with bounding boxes and confidence scores:

![Alt text](Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/val_batch2_pred.jpg)

#### Training Curves
Loss and metric progression across all 200 epochs:

![Alt text](Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/results.png)

## Deployment & Usage

### Getting the dataset & weights

The trained detector (`best.pt`, 21 MB) **is** committed, so inference and the
web app work straight after cloning. The training data is not -- 166 MB of
imagery is not this repository's to redistribute:

```bash
# Download from Roboflow, then unpack so the layout is:
#   Model-Training/Dataset/{train,valid,test}/{images,labels}/
# https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
```

Only `evaluate.py` and retraining need the dataset. To regenerate the ONNX
export used by the benchmark:

```bash
yolo export model=Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt format=onnx
```

### Install

```bash
# PyTorch first, matched to your CUDA version (see requirements-long.txt)
pip install -r requirements.txt
```

### Run the monitoring service

```bash
cd Model-Deployment
export PPE_SECRET_KEY="$(python -c 'import os;print(os.urandom(24).hex())')"
python app.py           # http://localhost:3333
```

The dashboard shows one card per tracked worker with their ID, the PPE they are
wearing, what they are missing, and live FPS / per-frame latency. Violations are
written to `detections.db` with an annotated snapshot and are browsable at
`/review`.

### Run in Docker

```bash
docker build -t ppe-detection .
docker run -p 3333:3333 -e PPE_SECRET_KEY=... ppe-detection
```

The image is CPU-only by default; for GPU inference swap the base image for
`nvidia/cuda:12.4.1-runtime-ubuntu22.04`, install the CUDA torch wheel and run
with `--gpus all`.

### Run the tests

```bash
./run_tests.sh
```

40 tests covering association geometry and the temporal state machine. They
import no torch, OpenCV or numpy and need no weights, so they run in
milliseconds in CI.

### Evaluate and benchmark

```bash
cd Model-Testing

# Held-out test split, per-class breakdown
python evaluate.py --split test --markdown ../metrics_test.md

# Latency percentiles and FPS, PyTorch vs ONNX
python benchmark.py --formats pt onnx --runs 200 --markdown ../latency.md

# Raw detector sanity checks
python checker.py        # CUDA/device report
python test_cam.py       # webcam, detections only
python test_single.py    # one image
```

### Retrain

```
Model-Training/yolov8-finetuning-for-ppe-detection.ipynb
```

## Dataset Information

**Construction Site Safety Image Dataset** sourced from [Roboflow](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety):
- **Total Images**: 2,801 annotated images
- **Classes**: 10 categories including PPE and non-PPE objects
- **Annotation Format**: YOLO format with normalized coordinates
- **Diversity**: Wide variety of construction site scenarios, lighting conditions, and worker configurations
- **Quality**: Professional annotations with high inter-rater agreement

## Limitations

Stated plainly, because they bound what the system can honestly claim:

- **Region priors assume upright, roughly frontal workers.** A crouching or
  prone worker can push a correctly worn helmet outside the expected band. The
  priors are constants in `ppe_association.py` and can be swept against a
  labelled set rather than hand-tuned.
- **Association is 2D and monocular.** With significant depth separation, a
  helmet on a distant worker can overlap a nearer person's box. Multi-view or
  depth input would resolve it; a single fixed camera cannot.
- **No pose estimation.** Keypoints would place the head and torso far more
  reliably than a box-fraction prior, at extra inference cost.
- **The test split is 82 images.** Metrics from it carry wide confidence
  intervals; treat per-class numbers as indicative, not precise.
- **Dataset domain.** Roboflow CSS imagery skews toward daylight and clear
  weather; night, rain, and high-visibility-clothing variation are
  under-represented.
- **Single camera, single process.** Multi-camera scaling would need the
  inference loop moved off the Flask worker onto a queue.

## Future Enhancements

1. **Pose-based association**: Replace vertical-band priors with YOLO-pose
   keypoints, so helmet checks anchor to the actual head position
2. **Sweep association thresholds**: Tune `MIN_CONTAINMENT` and the region bands
   against a labelled compliance set instead of setting them by hand
3. **Quantization**: INT8 / TensorRT export, reporting the mAP cost alongside
   the latency gain
4. **Model comparison**: YOLOv8n / v8m / v11 under identical settings, as an
   accuracy-vs-latency curve rather than a single point
5. **Multi-camera support**: Move inference onto a worker queue so one process
   can serve several feeds
6. **Edge deployment**: Jetson / Raspberry Pi with the quantized export
7. **Alert integrations**: Email, SMS or webhook fan-out on confirmed violations

## Attribution

The original PPE detection project, dataset preparation and initial Flask
application are the work of upstream authors and are used under the MIT License
(see [LICENSE](LICENSE)). Dataset: [Construction Site Safety Image
Dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
by Roboflow.

Work added in this fork: the person-to-PPE association module, ByteTrack
integration and per-track temporal state machine, the compliance visualization
layer, the 40-test suite, the test-split evaluation and latency benchmarking
scripts, the SQLite migration, and containerization.
