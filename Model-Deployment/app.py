"""SafeSite monitoring service: Flask + Socket.IO.

Three ways to feed frames in, one compliance pipeline behind them:

    browser camera / video file -> Socket.IO 'client_frame'   (works in the cloud)
    single image upload         -> POST /api/detect           (stateless)
    site camera / RTSP / file   -> MJPEG /video_feed          (on-prem)

Each video source gets its own MonitorSession -- its own tracker, debounce
state and rate limits -- so two people using the public demo never share
worker IDs, violation timers or alerts.

Configuration is by environment variable; see README "Configuration".
"""

import os

# Cap decoded image size before OpenCV is imported: an 8 MB upload can be a
# JPEG that decodes to gigabytes.
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', str(50_000_000))

import base64  # noqa: E402
import functools  # noqa: E402
import hmac  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections import deque  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import (Flask, Response, abort, jsonify, render_template, request,  # noqa: E402
                   session)
from flask_socketio import SocketIO  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from werkzeug.middleware.proxy_fix import ProxyFix  # noqa: E402

import visualization  # noqa: E402
from database import Database, stored_snapshot_path  # noqa: E402
from detection_logic import (ComplianceChecker, InstanceDetector, SerialCounter,  # noqa: E402
                             SnapshotManager, evaluate_still)
from settings_schema import (SettingsError, load_settings, save_settings,  # noqa: E402
                             validate_settings)
from storage import BlobSnapshotStore, LocalSnapshotStore  # noqa: E402


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None or value.strip() == '':
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.environ.get('PPE_DATA_DIR', BASE_DIR))
os.makedirs(DATA_DIR, exist_ok=True)

# Public demo: visitors stream from their own browser, see only their own
# violations, and cannot change settings or touch the site camera.
PUBLIC_DEMO = env_bool('PPE_PUBLIC_DEMO')
ADMIN_PASSWORD = os.environ.get('PPE_ADMIN_PASSWORD', '')
SERVER_CAMERA = env_bool('PPE_SERVER_CAMERA', default=not PUBLIC_DEMO)
VIDEO_SOURCE = os.environ.get('PPE_VIDEO_SOURCE', '0')
MAX_SESSIONS = int(os.environ.get('PPE_MAX_SESSIONS', '4'))
RETENTION_HOURS = float(os.environ.get('PPE_RETENTION_HOURS', '0'))

SESSION_IDLE_SECONDS = 60       # a browser session with no frames this long can be reclaimed
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_FRAME_SIDE = 1280           # larger inputs are downscaled before inference

SETTINGS_FILE = os.path.join(DATA_DIR, 'settings.json')

# Persistent storage (the Azure deployment). Unset = SQLite + local folder.
#   PPE_DATABASE_URL      libpq conninfo or URL, e.g. "host=... dbname=... user=... sslmode=require"
#   PPE_DB_AUTH=entra     sign in with a Microsoft Entra token (managed identity), no password
#   PPE_BLOB_ACCOUNT_URL  e.g. https://<account>.blob.core.windows.net
DATABASE_URL = os.environ.get('PPE_DATABASE_URL', '')
DB_AUTH = os.environ.get('PPE_DB_AUTH', '').strip().lower()
BLOB_ACCOUNT_URL = os.environ.get('PPE_BLOB_ACCOUNT_URL', '')
BLOB_CONTAINER = os.environ.get('PPE_BLOB_CONTAINER', 'snapshots')
POSTGRES_TOKEN_SCOPE = 'https://ossrdbms-aad.database.windows.net/.default'

# Weights are searched best-first so the app never silently serves a weaker
# model than the one the results in the README were measured on.
_RUNS = os.path.join(BASE_DIR, '..', 'Model-Training', 'Outputs', 'runs', 'detect')
MODEL_CANDIDATES = [
    os.path.join(_RUNS, 'yolov8s_ppe_css_200_epochs', 'weights', 'best.pt'),
    os.path.join(_RUNS, 'yolov8s_ppe_css_80_epochs', 'weights', 'best.pt'),
]


def resolve_model_path():
    """The weights to serve. An explicit PPE_MODEL_PATH that is missing is fatal."""
    explicit = os.environ.get('PPE_MODEL_PATH')
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"PPE_MODEL_PATH={explicit!r} does not exist")
        return os.path.abspath(explicit)
    return next((os.path.abspath(p) for p in MODEL_CANDIDATES if os.path.exists(p)), None)


MODEL_PATH = resolve_model_path()


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

app = Flask(__name__)
app.config.update(
    # Random per process unless set; a fixed default would let anyone forge sessions.
    SECRET_KEY=os.environ.get('PPE_SECRET_KEY') or os.urandom(24).hex(),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=env_bool('PPE_SECURE_COOKIES'),
)

# No cors_allowed_origins: Socket.IO then accepts only same-origin pages, so
# another site cannot open a socket under a visitor's cookie.
socketio = SocketIO(app, async_mode='threading',
                    max_http_buffer_size=MAX_FRAME_BYTES)

# Behind Azure Container Apps' TLS-terminating ingress; trust one proxy hop.
# Applied after SocketIO() so it wraps the Socket.IO middleware too.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)



def build_storage():
    """Snapshot store + database, from the environment.

    In Azure both sign in as the app's managed identity (DefaultAzureCredential
    picks it up from AZURE_CLIENT_ID): no connection-string password and no
    storage account key exists anywhere in the deployment.
    """
    credential = None
    if BLOB_ACCOUNT_URL or DB_AUTH == 'entra':
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()

    if BLOB_ACCOUNT_URL:
        store = BlobSnapshotStore(BLOB_ACCOUNT_URL, BLOB_CONTAINER, credential)
    else:
        store = LocalSnapshotStore(os.path.join(DATA_DIR, 'snapshots'))

    token_provider = None
    if DATABASE_URL and DB_AUTH == 'entra':
        def token_provider():
            return credential.get_token(POSTGRES_TOKEN_SCOPE).token

    database = Database(os.path.join(DATA_DIR, 'detections.db'), url=DATABASE_URL or None,
                        token_provider=token_provider, store=store)
    database.init_db()
    return store, database


snapshot_store, db = build_storage()
snapshot_manager = SnapshotManager(store=snapshot_store)
# Floor from the database: a restarted container must not reuse stored IDs.
serial_counter = SerialCounter(os.path.join(DATA_DIR, 'instance_counter.json'),
                               floor=db.max_serial)
compliance_checker = ComplianceChecker()

current_settings = load_settings(SETTINGS_FILE)
dev_mode = False

# One model for stateless image requests; each video source loads its own,
# because ByteTrack state lives on the model object (persist=True).
image_model = None
# Torch already spreads one inference across all cores; running two at once
# only makes both slower. Sessions take turns frame by frame.
inference_lock = threading.Lock()


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------

def new_model():
    """Load a fresh model instance from the resolved weights."""
    if MODEL_PATH is None:
        print("[app] Fine-tuned weights not found - falling back to yolov8n.pt "
              "(COCO classes only, PPE detection will not work)")
        return YOLO('yolov8n.pt')
    return YOLO(MODEL_PATH)


def run_inference(model, frame, settings, track=True):
    """Detect people and PPE in one frame, with track IDs when tracking is on.

    Tracking is what makes per-worker debouncing possible: ByteTrack keeps a
    stable ID on each worker across frames, so a violation timer belongs to a
    person rather than to the frame, and one worker cannot be logged twice.
    """
    conf = settings.get('confidence_threshold', 0.35)

    with inference_lock:
        if track and settings.get('enable_tracking', True):
            results = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                  conf=conf, verbose=False)
        else:
            results = model.predict(frame, conf=conf, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        ids = getattr(boxes, 'id', None)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                'class': model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'track_id': int(ids[i].item()) if ids is not None else None,
            })

    return detections


def decode_image(data):
    """Bytes -> BGR frame, downscaled to MAX_FRAME_SIDE. None if undecodable."""
    if not data:
        return None
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    scale = MAX_FRAME_SIDE / max(height, width)
    if scale < 1:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)),
                           interpolation=cv2.INTER_AREA)
    return frame


def utc_now():
    """Timezone-aware timestamp, so the browser converts it to local time correctly."""
    return datetime.now(timezone.utc).isoformat()


class MonitorSession:
    """Everything one video source needs: tracker, debounce state, rate limits."""

    def __init__(self, model, owner=None):
        self.model = model
        self.owner = owner                      # visitor ID on the public demo, else None
        self.detector = InstanceDetector(counter=serial_counter)
        self.detector.update_settings(current_settings)
        self.lock = threading.Lock()            # one frame in flight per source
        self.frame_times = deque(maxlen=30)     # rolling window for FPS
        self.inference_times = deque(maxlen=30)
        self.last_snapshot_time = {}
        self.last_alert_time = 0.0
        self.last_active = time.time()

    def fps(self):
        if len(self.frame_times) < 2:
            return 0.0
        span = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / span if span > 0 else 0.0

    def process(self, frame):
        """Detect -> associate -> debounce -> log one frame.

        Returns (annotated_frame, update, alert_or_None); the caller decides
        whether those go to one browser or to every dashboard.
        """
        settings = current_settings

        started = time.time()
        detections = run_inference(self.model, frame, settings)
        self.inference_times.append((time.time() - started) * 1000)

        result = self.detector.process_detection(detections, dev_mode, settings)
        is_compliant = compliance_checker.check_compliance(result, dev_mode)

        now = time.time()
        self.frame_times.append(now)
        self.last_active = now

        mean_ms = sum(self.inference_times) / len(self.inference_times)
        fps = self.fps()
        annotated = visualization.annotate(frame, result, fps=fps, inference_ms=mean_ms)

        self._log_snapshots(annotated, result, now, settings)
        alert = self._maybe_alert(result, is_compliant, now, settings)

        update = {
            'timestamp': utc_now(),
            'is_compliant': is_compliant,
            'detection_details': result,
            'fps': round(fps, 1),
            'inference_ms': round(mean_ms, 1),
            'dev_mode': dev_mode,
        }
        return annotated, update, alert

    def _log_snapshots(self, annotated, result, now, settings):
        """Snapshot each violating worker, rate-limited per instance so one
        worker cannot flood the review queue."""
        interval = settings.get('instance_reset_timeout', 5)

        for violation in result['violations']:
            instance_id = violation['instance_id']
            if now - self.last_snapshot_time.get(instance_id, 0) < interval:
                continue

            filename = self.detector.get_next_snapshot_filename(instance_id)
            if not filename:
                continue

            saved = snapshot_manager.save_snapshot(annotated, filename)
            if saved:
                db.log_instance_snapshot(
                    instance_id=instance_id,
                    missing_ppe=violation['missing_ppe'],
                    detected_ppe=violation['worn'],
                    snapshot_path=stored_snapshot_path(saved),
                    track_id=violation['track_id'],
                    owner=self.owner,
                )
                self.last_snapshot_time[instance_id] = now

    def _maybe_alert(self, result, is_compliant, now, settings):
        if is_compliant or not result['has_person']:
            return None
        if now - self.last_alert_time <= settings.get('non_compliance_delay', 3):
            return None

        description = (f"{result['violation_count']} worker(s) missing "
                       f"{', '.join(result['missing_ppe'])}")
        db.log_alert("NON_COMPLIANCE", description, None, owner=self.owner)
        self.last_alert_time = now

        return {
            'timestamp': utc_now(),
            'type': 'NON_COMPLIANCE',
            'description': description,
            'violations': result['violations'],
            'dev_mode': dev_mode,
        }


# ----------------------------------------------------------------------
# Access control
# ----------------------------------------------------------------------

def is_admin():
    """True for a request carrying the admin password.

    With no password configured, an on-prem install stays open (the previous
    behaviour) while the public demo has no admin at all.
    """
    if not ADMIN_PASSWORD:
        return not PUBLIC_DEMO
    auth = request.authorization
    if not auth or auth.username != 'admin' or auth.password is None:
        return False
    return hmac.compare_digest(auth.password.encode(), ADMIN_PASSWORD.encode())


def admin_denied():
    if ADMIN_PASSWORD:
        return (jsonify(status='error', message='Admin login required'), 401,
                {'WWW-Authenticate': 'Basic realm="SafeSite admin"'})
    return jsonify(status='error', message='Read-only on the public demo'), 403


def admin_required(view):
    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        if not is_admin():
            return admin_denied()
        return view(*args, **kwargs)
    return wrapper


def visitor_id():
    """Stable anonymous ID for this browser, kept in the signed session cookie."""
    if 'uid' not in session:
        session['uid'] = uuid.uuid4().hex
        session.permanent = True
    return session['uid']


def owner_scope():
    """Owner filter for reads: on the public demo a visitor sees only their own rows."""
    if not PUBLIC_DEMO or is_admin():
        return None
    return visitor_id()


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------

def page_context():
    return {
        'public_demo': PUBLIC_DEMO,
        'server_camera': SERVER_CAMERA,
        'retention_hours': RETENTION_HOURS,
        'read_only': PUBLIC_DEMO and not ADMIN_PASSWORD,
    }


@app.route('/')
def index():
    visitor_id()  # issue the cookie before the socket connects
    return render_template('index.html', **page_context())


@app.route('/review')
def review():
    # ?all=1 is the admin view. Challenging here, at a top-level path, makes
    # the browser send the credentials to every path afterwards -- including
    # /snapshots/, which a challenge under /api/ would not cover.
    if request.args.get('all') == '1' and not is_admin():
        return admin_denied()
    visitor_id()
    return render_template('review.html', admin_view=request.args.get('all') == '1',
                           **page_context())


@app.route('/settings')
def settings():
    return render_template('settings.html', **page_context())


@app.route('/healthz')
def healthz():
    """Liveness/readiness probe: 200 once the model is loaded and warmed up."""
    if image_model is None:
        return jsonify(status='starting'), 503
    return jsonify(status='ok',
                   model=os.path.basename(MODEL_PATH) if MODEL_PATH else 'yolov8n.pt',
                   storage={'database': db.dialect, 'snapshots': snapshot_store.kind},
                   browser_sessions=len(browser_sessions),
                   max_sessions=MAX_SESSIONS)


# ----------------------------------------------------------------------
# Settings API
# ----------------------------------------------------------------------

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(current_settings)


@app.route('/api/settings', methods=['POST'])
@admin_required
def update_settings():
    global current_settings
    try:
        new_settings = validate_settings(request.get_json(silent=True), base=current_settings)
    except SettingsError as e:
        return jsonify(status='error', message=str(e)), 400

    current_settings = new_settings
    if save_settings(SETTINGS_FILE, new_settings):
        return jsonify(status='success', message='Settings saved')
    return jsonify(status='error', message='Settings applied but could not be written to disk'), 500


@app.route('/api/settings/reset', methods=['POST'])
@admin_required
def reset_settings():
    global current_settings
    current_settings = validate_settings({})
    if save_settings(SETTINGS_FILE, current_settings):
        return jsonify(status='success', message='Settings reset to defaults',
                       settings=current_settings)
    return jsonify(status='error', message='Failed to save settings to file'), 500


@app.route('/toggle_dev_mode', methods=['POST'])
@admin_required
def toggle_dev_mode():
    global dev_mode
    dev_mode = not dev_mode
    return jsonify(status='success', dev_mode=dev_mode,
                   message=f'Dev mode {"enabled" if dev_mode else "disabled"}')


# ----------------------------------------------------------------------
# Single-image detection
# ----------------------------------------------------------------------

@app.route('/api/detect', methods=['POST'])
def detect_image():
    """Run the compliance check on one uploaded image. Nothing is stored."""
    upload = request.files.get('image')
    if upload is None:
        return jsonify(status='error', message="Send the image as multipart field 'image'"), 400

    frame = decode_image(upload.read())
    if frame is None:
        return jsonify(status='error', message='Unsupported or corrupt image'), 400

    settings = current_settings
    started = time.time()
    detections = run_inference(image_model, frame, settings, track=False)
    inference_ms = (time.time() - started) * 1000

    result = evaluate_still(detections, settings['required_ppe'],
                            strict=settings.get('compliance_mode') != 'explicit')
    annotated = visualization.annotate(frame, result, fps=None, inference_ms=inference_ms)

    ok, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return jsonify(status='error', message='Could not encode result'), 500

    return jsonify(
        status='success',
        image='data:image/jpeg;base64,' + base64.b64encode(buffer.tobytes()).decode('ascii'),
        update={
            'timestamp': utc_now(),
            'is_compliant': result['is_compliant'],
            'detection_details': result,
            'fps': None,
            'inference_ms': round(inference_ms, 1),
        },
    )


@app.errorhandler(413)
def too_large(_error):
    return jsonify(status='error',
                   message=f'Upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB'), 413


# ----------------------------------------------------------------------
# Browser-fed video (webcam or local video file) over Socket.IO
# ----------------------------------------------------------------------

browser_sessions = {}           # Socket.IO sid -> MonitorSession (None while loading)
sessions_lock = threading.Lock()


def _reclaim_idle_sessions():
    """Free sessions whose tab stopped sending frames. Caller holds sessions_lock."""
    cutoff = time.time() - SESSION_IDLE_SECONDS
    for sid in [s for s, sess in browser_sessions.items()
                if sess is not None and sess.last_active < cutoff]:
        del browser_sessions[sid]
        print(f"[app] Reclaimed idle browser session {sid}")


@socketio.on('client_start')
def on_client_start():
    """Reserve a slot and give this browser its own tracker and detector."""
    sid = request.sid
    owner = (session.get('uid') or uuid.uuid4().hex) if PUBLIC_DEMO else None

    with sessions_lock:
        if browser_sessions.get(sid) is not None:
            return {'ok': True}
        if len(browser_sessions) >= MAX_SESSIONS:
            _reclaim_idle_sessions()
        if len(browser_sessions) >= MAX_SESSIONS:
            return {'ok': False,
                    'error': 'The demo is at capacity right now - please try again in a minute.'}
        browser_sessions[sid] = None    # reserve while the model loads

    try:
        sess = MonitorSession(new_model(), owner=owner)
    except Exception as e:  # noqa: BLE001 - report any load failure to the client
        with sessions_lock:
            browser_sessions.pop(sid, None)
        print(f"[app] Could not start browser session: {e}")
        return {'ok': False, 'error': 'Could not load the model on the server.'}

    with sessions_lock:
        if sid not in browser_sessions:     # client left while we were loading
            return {'ok': False, 'error': 'Session closed'}
        browser_sessions[sid] = sess
    return {'ok': True}


@socketio.on('client_frame')
def on_client_frame(data):
    """One JPEG frame in, the annotated frame plus the compliance result out.

    The browser sends its next frame only after this returns, which is the
    backpressure: a slow server lowers the frame rate instead of queueing.
    """
    sess = browser_sessions.get(request.sid)
    if sess is None:
        return {'ok': False, 'error': 'not_started'}
    if not isinstance(data, (bytes, bytearray)) or len(data) > MAX_FRAME_BYTES:
        return {'ok': False, 'error': 'Frames must be JPEG bytes under 2 MB'}
    if not sess.lock.acquire(blocking=False):
        return {'ok': False, 'busy': True}

    try:
        frame = decode_image(bytes(data))
        if frame is None:
            return {'ok': False, 'error': 'Could not decode frame'}

        annotated, update, alert = sess.process(frame)
        ok, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return {'ok': False, 'error': 'Could not encode result'}
        return {'ok': True, 'frame': buffer.tobytes(), 'update': update, 'alert': alert}
    except Exception as e:  # noqa: BLE001 - keep the socket alive, report the failure
        print(f"[app] Error processing browser frame: {e}")
        return {'ok': False, 'error': 'Server error while processing the frame'}
    finally:
        sess.lock.release()


@socketio.on('client_stop')
def on_client_stop():
    with sessions_lock:
        browser_sessions.pop(request.sid, None)
    return {'ok': True}


@socketio.on('disconnect')
def on_disconnect(*_args):
    with sessions_lock:
        browser_sessions.pop(request.sid, None)


# ----------------------------------------------------------------------
# Site camera (on-prem): webcam index, RTSP URL or video file on the server
# ----------------------------------------------------------------------

camera = None
streaming = False
stream_lock = threading.Lock()
server_session = None


def open_video_source():
    source = int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video source {VIDEO_SOURCE!r}")
    return capture


def generate_frames():
    """MJPEG stream of the site camera; results are broadcast to every dashboard."""
    is_file = not VIDEO_SOURCE.isdigit() and os.path.isfile(VIDEO_SOURCE)

    while streaming:
        try:
            with stream_lock:
                if camera is None or not camera.isOpened():
                    break
                success, frame = camera.read()
                if not success and is_file:
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop demo videos

            if not success or frame is None:
                time.sleep(0.1)
                continue

            annotated, update, alert = server_session.process(frame)
            if alert:
                socketio.emit('alert', alert)
            socketio.emit('detection_update', update)

            ok, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except GeneratorExit:
            break
        except Exception as e:  # noqa: BLE001 - end this stream, keep the service up
            print(f"Error in generate_frames: {e}")
            break


@app.route('/video_feed')
def video_feed():
    if not SERVER_CAMERA or server_session is None:
        return jsonify(status='error', message='Site camera is not streaming'), 404
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_stream', methods=['POST'])
@admin_required
def start_stream():
    global camera, streaming, server_session

    if not SERVER_CAMERA:
        return jsonify(status='error',
                       message='No site camera on this server - use your browser camera instead.'), 403
    try:
        if server_session is None:
            server_session = MonitorSession(new_model())
        with stream_lock:
            if camera is None or not camera.isOpened():
                camera = open_video_source()
        streaming = True
        return jsonify(status='success', message='Stream started')
    except Exception as e:  # noqa: BLE001
        return jsonify(status='error', message=str(e)), 500


@app.route('/stop_stream', methods=['POST'])
@admin_required
def stop_stream():
    global camera, streaming

    streaming = False
    time.sleep(0.3)
    with stream_lock:
        if camera is not None:
            camera.release()
            camera = None
    return jsonify(status='success', message='Stream stopped')


# ----------------------------------------------------------------------
# Violation log API
# ----------------------------------------------------------------------

@app.route('/stats')
def get_stats():
    stats = db.get_statistics(owner=owner_scope())
    stats['dev_mode'] = dev_mode
    return jsonify(stats)


@app.route('/api/instances')
def get_instances():
    if request.args.get('all') == '1' and not is_admin():
        return admin_denied()
    try:
        instances = db.get_all_instances(request.args.get('sort', 'first_detected'),
                                         request.args.get('order', 'desc'),
                                         owner=owner_scope())
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(instances)


@app.route('/api/instance/<instance_id>/snapshots')
def get_instance_snapshots(instance_id):
    data = db.get_instance_snapshots(instance_id, owner=owner_scope())
    if data:
        return jsonify(data)
    return jsonify(error='Instance not found'), 404


def _send_snapshot(filename, as_attachment):
    """Serve one snapshot by bare file name, never by path.

    Only names recorded in the database are served (the store also strips any
    directory part), and on the public demo the snapshot must belong to this
    visitor. Blob storage stays private: images reach the browser only here.
    """
    name = os.path.basename(filename)
    exists, owner = db.snapshot_owner(name)
    if not exists:
        abort(404)
    scope = owner_scope()
    if scope is not None and owner != scope:
        abort(404)

    data = snapshot_store.load(name)
    if data is None:
        abort(404)
    response = Response(data, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'private, max-age=3600'
    if as_attachment:
        response.headers['Content-Disposition'] = f'attachment; filename="{name}"'
    return response


@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    return _send_snapshot(filename, as_attachment=False)


@app.route('/download_snapshot/<path:filename>')
def download_snapshot(filename):
    return _send_snapshot(filename, as_attachment=True)


@app.route('/api/delete_instance/<instance_id>', methods=['DELETE'])
def delete_instance(instance_id):
    # A demo visitor may delete their own rows; everything else needs admin.
    if PUBLIC_DEMO and not is_admin():
        owner = visitor_id()
    elif is_admin():
        owner = None
    else:
        return admin_denied()

    if db.delete_instance(instance_id, owner=owner):
        return jsonify(status='success', message='Instance deleted')
    return jsonify(error='Instance not found'), 404


# ----------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------

def _retention_loop():
    while True:
        time.sleep(600)
        removed = db.purge_older_than(RETENTION_HOURS)
        if removed:
            print(f"[app] Retention: removed {removed} instance(s) older than {RETENTION_HOURS}h")


def init_service():
    """Runs at import, so it happens under gunicorn as well as `python app.py`."""
    global image_model

    if MODEL_PATH:
        print(f"[app] Loading weights: {MODEL_PATH}")
    image_model = new_model()
    # Warm-up: the first inference pays for lazy initialisation; do it before
    # /healthz reports ready rather than on a visitor's first frame.
    run_inference(image_model, np.zeros((640, 640, 3), dtype=np.uint8),
                  current_settings, track=False)

    if RETENTION_HOURS > 0:
        threading.Thread(target=_retention_loop, name='retention', daemon=True).start()

    print(f"[app] Ready - public_demo={PUBLIC_DEMO} server_camera={SERVER_CAMERA} "
          f"database={db.dialect} snapshots={snapshot_store.kind} "
          f"data_dir={DATA_DIR} admin={'on' if ADMIN_PASSWORD else 'off'}")


init_service()


if __name__ == '__main__':
    # Development server. Production runs under gunicorn (see gunicorn.conf.py).
    socketio.run(app,
                 host=os.environ.get('HOST', '127.0.0.1'),
                 port=int(os.environ.get('PORT', '3333')),
                 debug=env_bool('PPE_DEBUG'),
                 allow_unsafe_werkzeug=True)
