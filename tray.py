"""
GenShape3D — 3090 Worker Tray
Wraps worker.py in a Windows system-tray icon so the worker
keeps running without an open terminal window.

Usage (normal):  python tray.py
Usage (startup): the Startup folder shortcut points here.
"""

import os
import sys
import subprocess
import threading
import time
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
WORKER_SCRIPT = HERE / "worker.py"
LOG_FILE = HERE / "tray-worker.log"
VENV_PYTHON = HERE / ".venv" / "Scripts" / "python.exe"

# Use the venv python if it exists, otherwise fall back to sys.executable
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# ── Logging ──────────────────────────────────────────────────────────────────
handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log = logging.getLogger("tray")
log.setLevel(logging.DEBUG)
log.addHandler(handler)

# ── State ────────────────────────────────────────────────────────────────────
proc = None          # worker subprocess
status = "starting"  # shown in tooltip
proc_lock = threading.Lock()

# ── Tray icon ─────────────────────────────────────────────────────────────────
def make_icon(color=(0, 200, 120)):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((4, 4, 60, 60), fill=color + (255,))
    return img

STATUS_COLORS = {
    "starting": (220, 160, 0),
    "idle":     (0, 200, 120),
    "working":  (0, 120, 255),
    "error":    (220, 50, 50),
    "stopped":  (140, 140, 140),
}

# ── Worker process management ─────────────────────────────────────────────────
def start_worker():
    global proc, status
    with proc_lock:
        if proc and proc.poll() is None:
            log.info("start_worker called but worker already running")
            return
        log.info("Launching worker.py")
        try:
            proc = subprocess.Popen(
                [PYTHON, str(WORKER_SCRIPT)],
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            status = "idle"
            log.info("Worker PID %d started", proc.pid)
        except Exception as e:
            status = "error"
            log.error("Failed to start worker: %s", e)

def stop_worker():
    global proc, status
    with proc_lock:
        if proc and proc.poll() is None:
            log.info("Stopping worker PID %d", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            status = "stopped"
            log.info("Worker stopped")

def pipe_reader():
    """Read worker stdout/stderr and forward to log. Also detect 'processing' keyword."""
    global status
    while True:
        with proc_lock:
            p = proc
        if p is None:
            time.sleep(1)
            continue
        line = p.stdout.readline()
        if line:
            line = line.rstrip()
            log.info("[worker] %s", line)
            low = line.lower()
            if any(k in low for k in ("claiming", "processing", "running", "progress")):
                status = "working"
            elif any(k in low for k in ("complete", "done", "idle", "waiting", "no job")):
                status = "idle"
            elif "error" in low or "exception" in low or "traceback" in low:
                status = "error"
        else:
            # EOF — process ended
            with proc_lock:
                if p.poll() is not None:
                    log.warning("Worker process exited with code %s — restarting in 5s", p.returncode)
                    status = "error"
                    time.sleep(5)
                    start_worker()
            time.sleep(0.2)

def watchdog():
    """Restart worker if it dies unexpectedly."""
    while True:
        time.sleep(10)
        with proc_lock:
            p = proc
            if p is not None and p.poll() is not None:
                log.warning("Watchdog: worker dead (code %s), restarting", p.returncode)
                start_worker()

# ── Tray menu actions ─────────────────────────────────────────────────────────
def on_show_log(icon, item):
    os.startfile(str(LOG_FILE))

def on_stop(icon, item):
    stop_worker()
    icon.icon = make_icon(STATUS_COLORS["stopped"])
    icon.title = "GenShape3D 3090 — stopped"

def on_quit(icon, item):
    stop_worker()
    icon.stop()

def update_icon(icon):
    """Periodically refresh the tray tooltip and colour."""
    while True:
        time.sleep(3)
        color = STATUS_COLORS.get(status, (140, 140, 140))
        icon.icon = make_icon(color)
        icon.title = f"GenShape3D 3090 — {status}"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("Tray starting")
    start_worker()

    threading.Thread(target=pipe_reader, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()

    menu = pystray.Menu(
        pystray.MenuItem("GenShape3D 3090 Worker", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Show Log", on_show_log),
        pystray.MenuItem("Stop Worker", on_stop),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    icon = pystray.Icon(
        "genshape3d-3090",
        make_icon(STATUS_COLORS["starting"]),
        "GenShape3D 3090 — starting",
        menu,
    )

    threading.Thread(target=update_icon, args=(icon,), daemon=True).start()
    icon.run()

if __name__ == "__main__":
    main()
