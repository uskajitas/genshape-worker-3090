"""
genshape-worker-3090 — multi-model GPU worker dispatcher.

Long-lived process that:
  1. Registers with the API server on startup (declares which models it can run).
  2. Long-polls /api/workers/<id>/claim for a job whose `model` matches.
  3. Downloads the input image to a temp dir.
  4. Spawns the matching runner subprocess (each model lives in its own venv
     under runners/<model>/, with its own dependencies — no cross-contamination).
  5. Streams JSONL events from the runner's stdout: progress events go back
     to /api/workers/<id>/progress; done/failed events terminate the job.
  6. On done, uploads the runner's output mesh to R2 and calls /complete with
     the resulting URL. On failed (or non-zero exit), calls /complete with
     status="failed".
  7. Heartbeats every 10s so the server can signal cancellation; on cancel,
     sends SIGTERM (then SIGKILL after 5s) to the runner subprocess.

This file deliberately has no torch / model dependencies. It just needs
`requests` and `boto3` (R2 upload). Each runner brings its own torch + model
deps in its own venv. Adding a new model is a matter of dropping a folder
under runners/ and updating the MODELS env var.

Env vars (see .env.example for the full list):
  WORKER_ID            — unique string, e.g. "win-3090"
  WORKER_MODELS        — comma-separated, e.g. "hunyuan3d,triposr,sf3d,hi3dgen"
  WORKER_CAPACITY      — concurrent jobs (default 1)
  WORKER_API_BASE      — e.g. "https://api.genshape3d.com"
  WORKER_AUTH_TOKEN    — shared secret matching the server's env
  R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_PUBLIC_URL
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import boto3
import requests
from dotenv import load_dotenv

# ─── Config ──────────────────────────────────────────────────────────────────
load_dotenv()

WORKER_ID = os.environ["WORKER_ID"]
WORKER_MODELS = [m.strip() for m in os.environ["WORKER_MODELS"].split(",") if m.strip()]
WORKER_CAPACITY = int(os.environ.get("WORKER_CAPACITY", "1"))
API_BASE = os.environ["WORKER_API_BASE"].rstrip("/")
AUTH_TOKEN = os.environ["WORKER_AUTH_TOKEN"]

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_BUCKET = os.environ["R2_BUCKET"]
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL") or f"{R2_ENDPOINT}/{R2_BUCKET}"

HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}

REPO_ROOT = Path(__file__).resolve().parent
RUNNERS_DIR = REPO_ROOT / "runners"

# Shared tray-visible state. Updated by the worker thread; read by the
# tray UI thread. Lock-free reads are fine — pystray polls every second.
_state = {
    "status": "starting",       # "starting" | "idle" | "claiming" | "working"
    "current_job": None,        # {"id": str, "model": str, "started": float}
    "jobs_done": 0,
    "jobs_failed": 0,
    "started_at": time.time(),
}

# Where each model's venv python lives. Bootstrapped by setup scripts;
# checked at startup so we fail fast if a runner isn't installed.
def runner_python(model: str) -> Path:
    # Windows venv layout: runners/<model>/.venv/Scripts/python.exe
    return RUNNERS_DIR / model / ".venv" / "Scripts" / "python.exe"

def runner_entrypoint(model: str) -> Path:
    return RUNNERS_DIR / model / "run.py"

# ─── HTTP helpers (retry on transient failures) ─────────────────────────────
def _post(path: str, body: dict, timeout: float = 30.0) -> requests.Response | None:
    url = f"{API_BASE}{path}"
    for attempt in range(5):
        try:
            r = requests.post(url, headers=HEADERS, json=body, timeout=timeout)
            return r
        except requests.RequestException as e:
            wait = min(2 ** attempt, 30)
            print(f"[http] {path} failed ({e}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None

# ─── R2 client ───────────────────────────────────────────────────────────────
_s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)

def upload_result(local_path: Path, content_type: str = "model/gltf-binary") -> str:
    key = f"outputs/{uuid.uuid4()}{local_path.suffix}"
    _s3.upload_file(str(local_path), R2_BUCKET, key, ExtraArgs={"ContentType": content_type})
    return f"{R2_PUBLIC_URL}/{key}"

# ─── Lifecycle: register + heartbeat ─────────────────────────────────────────
def register() -> None:
    body = {"id": WORKER_ID, "models": WORKER_MODELS, "capacity": WORKER_CAPACITY}
    r = _post("/api/workers/register", body)
    if r is None or not r.ok:
        raise RuntimeError(f"register failed: {r.status_code if r else 'no response'} {r.text if r else ''}")
    print(f"[worker] registered as {WORKER_ID} models={WORKER_MODELS} capacity={WORKER_CAPACITY}")

# Tracks job IDs currently being processed so the heartbeat thread knows
# what to ask the server about. Reads/writes are coarse — a Set + Lock
# is enough for capacity=1; would still be fine at capacity=4.
_active_jobs: dict[str, "JobProc"] = {}
_active_lock = threading.Lock()

class JobProc:
    """Wraps a running runner subprocess so the heartbeat thread can cancel it."""
    def __init__(self, job_id: str, popen: subprocess.Popen):
        self.job_id = job_id
        self.popen = popen
        self.cancelled = False

    def cancel(self) -> None:
        if self.popen.poll() is not None:
            return
        self.cancelled = True
        print(f"[worker] cancel requested for job {self.job_id} — SIGTERM")
        try:
            self.popen.terminate()
        except Exception as e:
            print(f"[worker] terminate failed: {e}", file=sys.stderr)
        # If it doesn't die in 5s, kill it.
        try:
            self.popen.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"[worker] job {self.job_id} ignored SIGTERM — SIGKILL")
            self.popen.kill()

def heartbeat_loop() -> None:
    while True:
        time.sleep(10)
        with _active_lock:
            ids = list(_active_jobs.keys())
        body = {"jobIds": ids}
        r = _post(f"/api/workers/{WORKER_ID}/heartbeat", body, timeout=15.0)
        if r is None or not r.ok:
            continue
        cancelled = (r.json() or {}).get("cancelled", [])
        if cancelled:
            with _active_lock:
                for jid in cancelled:
                    proc = _active_jobs.get(jid)
                    if proc and not proc.cancelled:
                        proc.cancel()

# ─── Job execution ───────────────────────────────────────────────────────────
def download_image(url: str, dest: Path) -> None:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)

def report_progress(job_id: str, **kwargs: Any) -> None:
    body = {"jobId": job_id, **{k: v for k, v in kwargs.items() if v is not None}}
    _post(f"/api/workers/{WORKER_ID}/progress", body, timeout=15.0)

def report_complete(job_id: str, status: str, result_url: str = "") -> None:
    body = {"jobId": job_id, "status": status, "resultUrl": result_url}
    _post(f"/api/workers/{WORKER_ID}/complete", body, timeout=30.0)

def run_job(job: dict) -> None:
    job_id = job["id"]
    model = (job.get("model") or "hunyuan3d").strip()
    _state["status"] = "working"
    _state["current_job"] = {"id": job_id, "model": model, "started": time.time()}
    if model not in WORKER_MODELS:
        print(f"[worker] BUG: server gave us model={model} we don't support", file=sys.stderr)
        report_complete(job_id, "failed")
        _state["jobs_failed"] += 1
        _state["current_job"] = None
        _state["status"] = "idle"
        return

    runner_py = runner_python(model)
    runner_entry = runner_entrypoint(model)
    if not runner_py.exists():
        print(f"[worker] runner venv missing: {runner_py}", file=sys.stderr)
        report_complete(job_id, "failed")
        return
    if not runner_entry.exists():
        print(f"[worker] runner entrypoint missing: {runner_entry}", file=sys.stderr)
        report_complete(job_id, "failed")
        return

    with tempfile.TemporaryDirectory(prefix=f"job-{job_id}-") as td:
        tmp = Path(td)
        image_path = tmp / "input.png"
        output_dir = tmp / "out"
        output_dir.mkdir()
        job_json_path = tmp / "job.json"
        job_json_path.write_text(json.dumps(job), encoding="utf-8")

        try:
            download_image(job["imageUrl"], image_path)
        except Exception as e:
            print(f"[worker] failed to fetch input image: {e}", file=sys.stderr)
            report_complete(job_id, "failed")
            return

        cmd = [
            str(runner_py),
            str(runner_entry),
            "--job-json", str(job_json_path),
            "--image-path", str(image_path),
            "--output-dir", str(output_dir),
        ]
        print(f"[worker] spawning runner for job {job_id} model={model}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,  # line-buffered
            cwd=str(RUNNERS_DIR / model),
        )

        jp = JobProc(job_id, proc)
        with _active_lock:
            _active_jobs[job_id] = jp

        result_filename: str | None = None
        runner_error: str | None = None

        # Forward stderr in a background thread so it doesn't block stdout.
        def _drain_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                print(f"[runner:{model}] {line.rstrip()}", file=sys.stderr)
        threading.Thread(target=_drain_stderr, daemon=True).start()

        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON stdout from runner — log but don't crash.
                print(f"[runner:{model}] {line}")
                continue
            t = ev.get("type")
            if t == "progress":
                report_progress(
                    job_id,
                    pct=ev.get("pct"),
                    phase=ev.get("phase"),
                    step=ev.get("step"),
                    total=ev.get("total"),
                )
            elif t == "done":
                result_filename = ev.get("filename")
            elif t == "failed":
                runner_error = ev.get("error", "runner reported failure")
            elif t == "log":
                print(f"[runner:{model}] {ev.get('msg', '')}")

        rc = proc.wait()
        with _active_lock:
            _active_jobs.pop(job_id, None)

        if jp.cancelled:
            report_complete(job_id, "cancelled")
            _state["jobs_failed"] += 1
            _state["current_job"] = None
            _state["status"] = "idle"
            return
        if rc != 0 or runner_error or not result_filename:
            print(f"[worker] runner failed: rc={rc} err={runner_error} filename={result_filename}", file=sys.stderr)
            report_complete(job_id, "failed")
            _state["jobs_failed"] += 1
            _state["current_job"] = None
            _state["status"] = "idle"
            return

        local = output_dir / result_filename
        if not local.exists():
            print(f"[worker] runner reported done but file missing: {local}", file=sys.stderr)
            report_complete(job_id, "failed")
            _state["jobs_failed"] += 1
            _state["current_job"] = None
            _state["status"] = "idle"
            return

        try:
            content_type = {
                ".glb": "model/gltf-binary",
                ".gltf": "model/gltf+json",
                ".obj": "text/plain",
                ".ply": "application/octet-stream",
                ".stl": "application/octet-stream",
            }.get(local.suffix.lower(), "application/octet-stream")
            url = upload_result(local, content_type)
        except Exception as e:
            print(f"[worker] R2 upload failed: {e}", file=sys.stderr)
            report_complete(job_id, "failed")
            _state["jobs_failed"] += 1
            _state["current_job"] = None
            _state["status"] = "idle"
            return

        report_complete(job_id, "done", url)
        print(f"[worker] job {job_id} done → {url}")
        _state["jobs_done"] += 1
        _state["current_job"] = None
        _state["status"] = "idle"

# ─── Main loop: claim then run ───────────────────────────────────────────────
def claim_loop() -> None:
    while True:
        r = _post(f"/api/workers/{WORKER_ID}/claim", {}, timeout=40.0)
        if r is None:
            time.sleep(5)
            continue
        if r.status_code == 204:
            # No work this round; immediately re-poll (server holds the
            # connection ~25s, so this isn't a tight loop).
            continue
        if r.status_code == 409:
            # We're at capacity according to the server — wait briefly.
            time.sleep(5)
            continue
        if r.status_code == 404:
            # Server forgot about us (probably restarted). Re-register.
            print("[worker] server doesn't know us; re-registering")
            try:
                register()
            except Exception as e:
                print(f"[worker] re-register failed: {e}", file=sys.stderr)
                time.sleep(10)
            continue
        if not r.ok:
            print(f"[worker] claim error {r.status_code}: {r.text[:200]}", file=sys.stderr)
            time.sleep(5)
            continue

        try:
            payload = r.json()
        except ValueError:
            print(f"[worker] claim returned non-json: {r.text[:200]}", file=sys.stderr)
            time.sleep(5)
            continue
        job = payload.get("job")
        if not job:
            continue
        try:
            run_job(job)
        except Exception as e:
            # Last-resort guard so one bad job doesn't kill the worker.
            print(f"[worker] uncaught error in run_job: {e}", file=sys.stderr)
            try:
                report_complete(job["id"], "failed")
            except Exception:
                pass

def claim_loop_with_state() -> None:
    """Wraps claim_loop so we can flip _state.status to 'idle' before each
    poll (so the tray shows 'idle' between jobs without the worker thread
    needing to know about state)."""
    _state["status"] = "idle"
    claim_loop()

# ─── Tray UI (pystray) ──────────────────────────────────────────────────────
# Worker logic runs in background threads; the tray icon owns the main
# thread (Windows requires GUI on main thread). pystray polls _state every
# second to update the tooltip / menu text.

def _make_icon_image(color: tuple[int, int, int]):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill=color, outline=(0, 0, 0, 255), width=2)
    return img

def _format_uptime(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"

def run_tray() -> None:
    import pystray
    import webbrowser

    icon_idle    = _make_icon_image((46, 160, 67))   # green
    icon_busy    = _make_icon_image((47, 129, 247))  # blue
    icon_error   = _make_icon_image((218, 54, 51))   # red
    icon_starting = _make_icon_image((158, 158, 158))  # grey

    def status_line() -> str:
        s = _state["status"]
        cj = _state["current_job"]
        if s == "working" and cj:
            elapsed = int(time.time() - cj["started"])
            return f"Working: {cj['model']} · job {cj['id'][:8]} · {elapsed}s"
        if s == "idle":
            return "Idle — waiting for jobs"
        if s == "starting":
            return "Starting..."
        return f"Status: {s}"

    def stats_line() -> str:
        return f"Done: {_state['jobs_done']}  ·  Failed: {_state['jobs_failed']}  ·  Up: {_format_uptime(time.time() - _state['started_at'])}"

    def on_dashboard(_icon, _item):
        webbrowser.open("https://genshape3d.com/dashboard")

    def on_open_log(_icon, _item):
        log_path = REPO_ROOT / "logs" / "worker.log"
        if log_path.exists():
            os.startfile(str(log_path))  # type: ignore[attr-defined]

    def on_open_repo(_icon, _item):
        os.startfile(str(REPO_ROOT))  # type: ignore[attr-defined]

    def on_quit(icon, _item):
        print("[worker] tray quit requested")
        with _active_lock:
            for jp in list(_active_jobs.values()):
                jp.cancel()
        icon.stop()
        # Force exit so background threads don't keep the process alive.
        os._exit(0)

    icon = pystray.Icon(
        "genshape-worker-3090",
        icon_idle,
        f"GenShape3D Worker ({WORKER_ID})",
        menu=pystray.Menu(
            pystray.MenuItem(lambda _: status_line(), None, enabled=False),
            pystray.MenuItem(lambda _: stats_line(), None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(f"Models: {', '.join(WORKER_MODELS)}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open dashboard", on_dashboard),
            pystray.MenuItem("Open log file", on_open_log),
            pystray.MenuItem("Open worker folder", on_open_repo),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit worker", on_quit),
        ),
    )

    # Background updater: refreshes the tooltip + icon color every second.
    def updater():
        last_status = None
        while True:
            time.sleep(1)
            try:
                cj = _state["current_job"]
                tip_status = status_line()
                tip_stats  = stats_line()
                icon.title = f"GenShape3D Worker ({WORKER_ID})\n{tip_status}\n{tip_stats}"
                # Swap icon color on status change.
                s = _state["status"]
                if s != last_status:
                    icon.icon = (
                        icon_busy     if s == "working"
                        else icon_idle     if s == "idle"
                        else icon_starting
                    )
                    last_status = s
            except Exception as e:
                print(f"[tray] updater error: {e}", file=sys.stderr)

    threading.Thread(target=updater, daemon=True).start()
    icon.run()  # blocks main thread

def main() -> None:
    if not WORKER_MODELS:
        print("WORKER_MODELS env var is empty", file=sys.stderr)
        sys.exit(1)
    try:
        register()
    except Exception as e:
        print(f"[worker] register failed: {e}", file=sys.stderr)
        # Don't sys.exit — tray will still come up so the user can see what's wrong.
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    threading.Thread(target=claim_loop_with_state, daemon=True).start()
    print(f"[worker] background threads started; entering tray loop")
    try:
        run_tray()
    except KeyboardInterrupt:
        print("[worker] interrupted; cancelling active jobs")
        with _active_lock:
            for jp in list(_active_jobs.values()):
                jp.cancel()

if __name__ == "__main__":
    main()
