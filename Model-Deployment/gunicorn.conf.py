"""Gunicorn settings for the container.

One worker, many threads, on purpose: Socket.IO sessions, ByteTrack state and
the per-visitor detectors all live in process memory, so a second worker (or a
second replica) would receive frames for sessions it has never seen. Threads
are cheap here because inference itself is serialised behind a lock and torch
releases the GIL while it computes.
"""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '3333')}"
workers = 1
worker_class = 'gthread'
threads = int(os.environ.get('GUNICORN_THREADS', '32'))

# gthread heartbeats from the main loop, so long-lived WebSocket/MJPEG
# connections are not killed by this; it only catches a wedged worker.
timeout = 120
graceful_timeout = 20

# Logs go to stdout/stderr, which Container Apps ships to Log Analytics.
accesslog = '-'
errorlog = '-'
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(M)sms'
