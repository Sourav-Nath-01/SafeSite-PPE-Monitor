from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import threading
import time
from collections import deque

from detection_logic import InstanceDetector, ComplianceChecker, SnapshotManager
from database import Database
import visualization
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('PPE_SECRET_KEY', os.urandom(24).hex())
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

model = None
camera = None
streaming = False
stream_lock = threading.Lock()
dev_mode = False
db = Database()
instance_detector = InstanceDetector()
compliance_checker = ComplianceChecker()
snapshot_manager = SnapshotManager()

# Rolling window of frame times; the dashboard reports throughput, which is the
# number that decides whether this runs on site hardware or only on a workstation.
frame_times = deque(maxlen=30)
inference_times = deque(maxlen=30)

SETTINGS_FILE = 'settings.json'
DEFAULT_SETTINGS = {
    'required_ppe': {
        'helmet': True,
        'vest': True,
        'mask': False
    },
    'non_compliance_delay': 3,
    'instance_reset_timeout': 5,
    # 'strict' flags a required item with no positive detection; 'explicit'
    # waits for an actual NO-* detection before calling it a violation.
    'compliance_mode': 'strict',
    'confidence_threshold': 0.35,
    'enable_tracking': True,
}

# Weights are searched best-first so the app never silently serves a weaker
# model than the one the results in the README were measured on.
MODEL_CANDIDATES = [
    '../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt',
    '../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_80_epochs/weights/best.pt',
]


def load_settings():
    """Load settings from file, filling in any keys the file predates."""
    settings = DEFAULT_SETTINGS.copy()
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings.update(json.load(f))
    except Exception as e:
        print(f"Error loading settings: {e}")
    return settings


def save_settings_to_file(settings):
    """Persist settings to disk."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False


current_settings = load_settings()


def load_model():
    """Load the fine-tuned PPE weights, falling back to stock YOLO."""
    global model

    model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    if model_path is None:
        print("[app] Fine-tuned weights not found - falling back to yolov8n.pt "
              "(COCO classes only, PPE detection will not work)")
        model = YOLO('yolov8n.pt')
        return

    print(f"[app] Loading weights: {model_path}")
    model = YOLO(model_path)


def run_inference(frame):
    """Detect people and PPE in one frame, with track IDs when enabled.

    Tracking is what makes per-worker debouncing possible: ByteTrack keeps a
    stable ID on each worker across frames, so a violation timer belongs to a
    person rather than to the frame, and one worker cannot be logged twice.
    """
    conf = current_settings.get('confidence_threshold', 0.35)

    if current_settings.get('enable_tracking', True):
        results = model.track(frame, persist=True, tracker='bytetrack.yaml',
                              conf=conf, verbose=False)
    else:
        results = model(frame, conf=conf, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            track_id = None
            if getattr(boxes, 'id', None) is not None:
                track_id = int(boxes.id[i].item())

            detections.append({
                'class': model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'track_id': track_id,
            })

    return detections


def current_fps():
    """Mean FPS over the rolling window."""
    if len(frame_times) < 2:
        return 0.0
    span = frame_times[-1] - frame_times[0]
    if span <= 0:
        return 0.0
    return (len(frame_times) - 1) / span


def generate_frames():
    """Stream annotated frames, logging one snapshot per confirmed violation."""
    global camera, streaming, model, current_settings

    last_alert_time = 0
    last_snapshot_time = {}

    while streaming:
        try:
            with stream_lock:
                if camera is None or not camera.isOpened():
                    break

                success, frame = camera.read()

            if not success or frame is None:
                time.sleep(0.1)
                continue

            inference_start = time.time()
            detections = run_inference(frame)
            inference_times.append((time.time() - inference_start) * 1000)

            instance_result = instance_detector.process_detection(
                detections, dev_mode, current_settings)
            is_compliant = compliance_checker.check_compliance(instance_result, dev_mode)

            frame_times.append(time.time())
            mean_inference_ms = sum(inference_times) / len(inference_times) if inference_times else None

            annotated_frame = visualization.annotate(
                frame, instance_result, fps=current_fps(), inference_ms=mean_inference_ms)

            current_time = time.time()
            snapshot_interval = current_settings.get('instance_reset_timeout', 5)

            # Snapshot each violating worker independently, rate-limited per
            # instance so a single worker cannot flood the review queue.
            for violation in instance_result['violations']:
                instance_id = violation['instance_id']
                if current_time - last_snapshot_time.get(instance_id, 0) < snapshot_interval:
                    continue

                filename = instance_detector.get_next_snapshot_filename(instance_id)
                if not filename:
                    continue

                snapshot_path = snapshot_manager.save_snapshot(annotated_frame, filename)
                if snapshot_path:
                    db.log_instance_snapshot(
                        instance_id=instance_id,
                        missing_ppe=violation['missing_ppe'],
                        detected_ppe=violation['worn'],
                        snapshot_path=snapshot_path,
                        track_id=violation['track_id'],
                    )
                    last_snapshot_time[instance_id] = current_time

            if not is_compliant and instance_result['has_person']:
                alert_cooldown = current_settings.get('non_compliance_delay', 3)
                if current_time - last_alert_time > alert_cooldown:
                    description = (f"{instance_result['violation_count']} worker(s) missing "
                                   f"{', '.join(instance_result['missing_ppe'])}")
                    db.log_alert("NON_COMPLIANCE", description, None)

                    socketio.emit('alert', {
                        'timestamp': datetime.now().isoformat(),
                        'type': 'NON_COMPLIANCE',
                        'description': description,
                        'violations': instance_result['violations'],
                        'dev_mode': dev_mode
                    })
                    last_alert_time = current_time

            socketio.emit('detection_update', {
                'timestamp': datetime.now().isoformat(),
                'is_compliant': is_compliant,
                'detection_details': instance_result,
                'fps': round(current_fps(), 1),
                'inference_ms': round(mean_inference_ms, 1) if mean_inference_ms else None,
                'dev_mode': dev_mode
            })

            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except GeneratorExit:
            break
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/review')
def review():
    """Review page for instances and snapshots"""
    return render_template('review.html')

@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html')

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
    return jsonify(current_settings)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings"""
    global current_settings
    try:
        new_settings = request.json
        current_settings = new_settings
        
        instance_detector.update_settings(new_settings)
        
        if save_settings_to_file(new_settings):
            return jsonify({'status': 'success', 'message': 'Settings saved'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save settings to file'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    """Reset settings to defaults"""
    global current_settings
    try:
        current_settings = DEFAULT_SETTINGS.copy()
        instance_detector.update_settings(current_settings)
        
        if save_settings_to_file(current_settings):
            return jsonify({'status': 'success', 'message': 'Settings reset to defaults', 'settings': current_settings})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save settings to file'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    try:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {e}")
        return '', 500

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start video streaming"""
    global camera, streaming, model
    
    try:
        if model is None:
            load_model()
        
        with stream_lock:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        streaming = True
        return jsonify({'status': 'success', 'message': 'Stream started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop video streaming"""
    global camera, streaming
    
    try:
        streaming = False
        time.sleep(0.3)
        
        with stream_lock:
            if camera is not None:
                camera.release()
                camera = None
        
        return jsonify({'status': 'success', 'message': 'Stream stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/toggle_dev_mode', methods=['POST'])
def toggle_dev_mode():
    """Toggle dev/testing mode"""
    global dev_mode
    dev_mode = not dev_mode
    return jsonify({'status': 'success', 'dev_mode': dev_mode, 
                   'message': f'Dev mode {"enabled" if dev_mode else "disabled"}'})

@app.route('/stats')
def get_stats():
    """Get detection statistics"""
    stats = db.get_statistics()
    stats['dev_mode'] = dev_mode
    return jsonify(stats)

@app.route('/api/instances')
def get_instances():
    """Get all detection instances"""
    try:
        sort_by = request.args.get('sort', 'first_detected')
        sort_order = request.args.get('order', 'desc')
        
        instances = db.get_all_instances(sort_by, sort_order)
        return jsonify(instances)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/instance/<instance_id>/snapshots')
def get_instance_snapshots(instance_id):
    """Get all snapshots for a specific instance"""
    try:
        data = db.get_instance_snapshots(instance_id)
        if data:
            return jsonify(data)
        return jsonify({'error': 'Instance not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_snapshot/<path:filename>')
def download_snapshot(filename):
    """Download snapshot"""
    try:
        filepath = filename if os.path.isabs(filename) else os.path.join('snapshots', filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    """Serve snapshot for viewing"""
    try:
        filepath = os.path.join('snapshots', filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/jpeg')
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_instance/<instance_id>', methods=['DELETE'])
def delete_instance(instance_id):
    """Delete an instance and its snapshot"""
    try:
        success = db.delete_instance(instance_id)
        if success:
            return jsonify({'status': 'success', 'message': 'Instance deleted'})
        return jsonify({'error': 'Failed to delete instance'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        db.init_db()
        load_model()
        instance_detector.update_settings(current_settings)
        socketio.run(app, debug=True, host='localhost', port=3333)
    except Exception as e:
        print(f"Fatal error starting app: {e}")
