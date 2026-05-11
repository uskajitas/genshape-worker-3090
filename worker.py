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

        cj_started = _state["current_job"]["started"] if _state["current_job"] else time.time()
        if jp.cancelled:
            report_complete(job_id, "cancelled")
            _state["jobs_failed"] += 1
            _record_completion(job_id, model, "cancelled", cj_started, "user cancelled")
            _state["current_job"] = None
            _state["status"] = "idle"
            return
        if rc != 0 or runner_error or not result_filename:
            print(f"[worker] runner failed: rc={rc} err={runner_error} filename={result_filename}", file=sys.stderr)
            report_complete(job_id, "failed")
            _state["jobs_failed"] += 1
            _record_completion(job_id, model, "failed", cj_started, runner_error or f"exit={rc}")
            _state["current_job"] = None
            _state["status"] = "idle"
            return

        local = output_dir / result_filename
        if not local.exists():
            print(f"[worker] runner reported done but file missing: {local}", file=sys.stderr)
            report_complete(job_id, "failed")
            _state["jobs_failed"] += 1
            _record_completion(job_id, model, "failed", cj_started, "result file missing")
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
            _record_completion(job_id, model, "failed", cj_started, f"R2 upload: {e}")
            _state["current_job"] = None
            _state["status"] = "idle"
            return

        report_complete(job_id, "done", url)
        print(f"[worker] job {job_id} done → {url}")
        _state["jobs_done"] += 1
        _record_completion(job_id, model, "done", cj_started)
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
    _state["status"] = "idle"
    claim_loop()

# ── Recent-jobs ring buffers (for the UI's history lists) ───────────────────
# We don't have direct DB access from the worker (by design — the worker
# talks to the server via HTTP). So the history shown in the UI is what
# THIS worker has handled, kept in memory. Server restarts → history clears.
_history_lock = threading.Lock()
_completed_jobs: list[dict] = []  # most recent first
_failed_jobs: list[dict] = []
_HISTORY_MAX = 20

def _record_completion(job_id: str, model: str, status: str, started: float, error: str = "") -> None:
    rec = {
        "id": job_id,
        "model": model,
        "status": status,
        "startedAt": started,
        "completedAt": time.time(),
        "error": error,
    }
    with _history_lock:
        target = _completed_jobs if status == "done" else _failed_jobs
        target.insert(0, rec)
        del target[_HISTORY_MAX:]

# ─── UI: local HTTP server + browser ────────────────────────────────────────
# Same architecture as the 1080's Electron app, but lighter: serve the same
# index.html (with the API calls swapped from window.api → fetch), open the
# user's default browser at it. No GUI library deps needed.

_nvml = None
_nvml_handle = None
_nvml_name = None

def _init_nvml() -> None:
    global _nvml, _nvml_handle, _nvml_name
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml = pynvml
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        n = pynvml.nvmlDeviceGetName(_nvml_handle)
        _nvml_name = n.decode("utf-8", errors="replace") if isinstance(n, bytes) else n
    except Exception as e:
        print(f"[ui] NVML unavailable: {e}", file=sys.stderr)

def _gpu_snapshot() -> dict:
    if not _nvml or not _nvml_handle:
        return {"name": "(GPU stats unavailable)"}
    try:
        mem = _nvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        util = _nvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
        temp = _nvml.nvmlDeviceGetTemperature(_nvml_handle, _nvml.NVML_TEMPERATURE_GPU)
        try: power_w = _nvml.nvmlDeviceGetPowerUsage(_nvml_handle) / 1000.0
        except Exception: power_w = None
        return {
            "name": _nvml_name,
            "vram_used_gb": mem.used / 1e9,
            "vram_total_gb": mem.total / 1e9,
            "vram_pct": (mem.used / mem.total) * 100 if mem.total else 0,
            "util_pct": util.gpu,
            "temp_c": temp,
            "power_w": power_w,
        }
    except Exception as e:
        return {"name": "(GPU read error)", "_error": str(e)}

def _build_state_json() -> bytes:
    cj = _state["current_job"]
    cj_payload = None
    if cj:
        cj_payload = {
            "id": cj["id"],
            "model": cj["model"],
            "startedAt": _epoch_iso(cj["started"]),
            "progress": {"pct": cj.get("pct"), "phase": cj.get("phase"), "detail": cj.get("phase")},
        }
    with _history_lock:
        completed = [
            {**j, "startedAt": _epoch_iso(j["startedAt"]), "completedAt": _epoch_iso(j["completedAt"])}
            for j in _completed_jobs
        ]
        failed = [
            {**j, "startedAt": _epoch_iso(j["startedAt"]), "completedAt": _epoch_iso(j["completedAt"])}
            for j in _failed_jobs if j.get("status") == "failed"
        ]
        cancelled = [
            {**j, "startedAt": _epoch_iso(j["startedAt"]), "completedAt": _epoch_iso(j["completedAt"])}
            for j in _failed_jobs if j.get("status") == "cancelled"
        ]
    payload = {
        "isProcessing": cj is not None,
        "currentJob": cj_payload,
        # We don't see other workers' jobs from here (we'd need DB or a server
        # endpoint). Show this worker's current as the only "processing" entry.
        "processingJobs": [{"id": cj["id"], "status": "processing",
                            "startedAt": _epoch_iso(cj["started"]),
                            "progressPct": cj.get("pct", 0),
                            "progressPhase": cj.get("phase", "")}] if cj else [],
        "pendingJobs": [],   # the server queue isn't visible from the worker
        "completedJobs": completed,
        "failedJobs": failed,
        "cancelledJobs": cancelled,
        "gpu": _gpu_snapshot(),
        "worker": {
            "id": WORKER_ID,
            "models": WORKER_MODELS,
            "capacity": WORKER_CAPACITY,
            "uptime_s": int(time.time() - _state["started_at"]),
            "jobs_done": _state["jobs_done"],
            "jobs_failed": _state["jobs_failed"],
        },
    }
    return json.dumps(payload).encode("utf-8")

def _epoch_iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))

def _make_handler():
    """Build a BaseHTTPRequestHandler bound to the index.html on disk."""
    from http.server import BaseHTTPRequestHandler
    index_path = REPO_ROOT / "ui" / "index.html"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence default access log
            pass

        def _send(self, code: int, ctype: str, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                try:
                    body = index_path.read_bytes()
                    return self._send(200, "text/html; charset=utf-8", body)
                except FileNotFoundError:
                    return self._send(500, "text/plain", b"index.html missing")
            if self.path == "/state":
                return self._send(200, "application/json", _build_state_json())
            return self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if self.path.startswith("/cancel/"):
                jid = self.path[len("/cancel/"):]
                with _active_lock:
                    jp = _active_jobs.get(jid)
                if jp:
                    jp.cancel()
                    return self._send(200, "application/json", b'{"ok":true}')
                return self._send(404, "application/json", b'{"ok":false,"error":"not active here"}')
            return self._send(404, "text/plain", b"not found")

    return Handler

def run_ui_server() -> None:
    """Start the local HTTP UI on a free port, then open the default browser
    to it. Blocks the main thread (serves forever)."""
    import http.server
    import socketserver
    import webbrowser

    port = int(os.environ.get("WORKER_UI_PORT", "8765"))
    Handler = _make_handler()

    # Bind to localhost only — never expose this to the network.
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    print(f"[ui] HTTP server on http://127.0.0.1:{port}/")

    # Open the browser tab once, in a background thread so it doesn't block
    # the server. If WORKER_UI_NO_OPEN=1 in env, skip (useful for dev).
    if os.environ.get("WORKER_UI_NO_OPEN") != "1":
        def _open():
            time.sleep(0.5)  # let the server come up
            try:
                webbrowser.open(f"http://127.0.0.1:{port}/")
            except Exception as e:
                print(f"[ui] couldn't open browser: {e}", file=sys.stderr)
        threading.Thread(target=_open, daemon=True).start()

    httpd.serve_forever()

# ── (legacy: kept for reference) ────────────────────────────────────────────
def _format_uptime(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s % 60:02d}s"

def _format_bytes(b: int) -> str:
    if b < 1024:        return f"{b} B"
    if b < 1024**2:     return f"{b/1024:.1f} KB"
    if b < 1024**3:     return f"{b/1024**2:.1f} MB"
    return f"{b/1024**3:.2f} GB"

def run_gui_unused() -> None:
    import tkinter as tk
    from tkinter import ttk
    import webbrowser

    # Try to load NVML for GPU stats. If it fails, the GPU section just
    # shows "unavailable" — worker still works without it.
    nvml_handle = None
    nvml_name = "(GPU stats unavailable)"
    try:
        import pynvml  # nvidia-ml-py installs this module name
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        nvml_name = pynvml.nvmlDeviceGetName(nvml_handle)
        if isinstance(nvml_name, bytes):
            nvml_name = nvml_name.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[gui] NVML unavailable: {e}", file=sys.stderr)

    root = tk.Tk()
    root.title(f"GenShape3D Worker ({WORKER_ID})")
    root.geometry("520x500")
    root.minsize(480, 460)

    # Style
    BG = "#0d1117"; FG = "#e6edf3"; MUTED = "#8b949e"
    GREEN = "#3fb950"; BLUE = "#2f81f7"; RED = "#f85149"; AMBER = "#d29922"
    root.configure(bg=BG)
    style = ttk.Style()
    try: style.theme_use("clam")
    except Exception: pass
    style.configure("TProgressbar", troughcolor="#21262d", background=BLUE, bordercolor=BG)

    def lbl(parent, text, **kw):
        kw.setdefault("fg", FG)
        kw.setdefault("bg", BG)
        return tk.Label(parent, text=text, **kw)

    # ── Header: status banner ─────────────────────────────────────────
    header = tk.Frame(root, bg=BG); header.pack(fill="x", padx=16, pady=(14, 6))
    status_dot = tk.Canvas(header, width=18, height=18, bg=BG, highlightthickness=0)
    status_dot.pack(side="left")
    dot_id = status_dot.create_oval(2, 2, 16, 16, fill=GREEN, outline="")
    status_text = lbl(header, "Idle", font=("Segoe UI", 14, "bold"))
    status_text.pack(side="left", padx=(8, 0))

    detail_text = lbl(root, "Waiting for jobs", font=("Segoe UI", 9), fg=MUTED)
    detail_text.pack(anchor="w", padx=18)

    # ── GPU box ───────────────────────────────────────────────────────
    gpu_frame = tk.LabelFrame(root, text=" GPU ", bg=BG, fg=FG, font=("Segoe UI", 9, "bold"),
                              labelanchor="nw", padx=12, pady=10, bd=1, relief="solid")
    gpu_frame.pack(fill="x", padx=14, pady=(12, 6))
    gpu_name_lbl = lbl(gpu_frame, nvml_name, font=("Segoe UI", 10))
    gpu_name_lbl.pack(anchor="w")

    vram_row = tk.Frame(gpu_frame, bg=BG); vram_row.pack(fill="x", pady=(8, 2))
    lbl(vram_row, "VRAM:", font=("Segoe UI", 9), fg=MUTED).pack(side="left")
    vram_text = lbl(vram_row, "—", font=("Segoe UI", 10, "bold"))
    vram_text.pack(side="left", padx=(6, 0))

    vram_bar = ttk.Progressbar(gpu_frame, length=460, mode="determinate", maximum=100)
    vram_bar.pack(fill="x", pady=(2, 8))

    misc_row = tk.Frame(gpu_frame, bg=BG); misc_row.pack(fill="x")
    util_text = lbl(misc_row, "GPU util: —", font=("Segoe UI", 9))
    util_text.pack(side="left")
    temp_text = lbl(misc_row, "Temp: —", font=("Segoe UI", 9))
    temp_text.pack(side="left", padx=(20, 0))
    power_text = lbl(misc_row, "Power: —", font=("Segoe UI", 9))
    power_text.pack(side="left", padx=(20, 0))

    # ── Current job ───────────────────────────────────────────────────
    job_frame = tk.LabelFrame(root, text=" Current job ", bg=BG, fg=FG,
                              font=("Segoe UI", 9, "bold"), labelanchor="nw",
                              padx=12, pady=10, bd=1, relief="solid")
    job_frame.pack(fill="x", padx=14, pady=6)
    job_text = lbl(job_frame, "(none)", font=("Segoe UI", 10), fg=MUTED, justify="left")
    job_text.pack(anchor="w")

    # ── Counters + Models ─────────────────────────────────────────────
    info_frame = tk.LabelFrame(root, text=" Worker ", bg=BG, fg=FG,
                               font=("Segoe UI", 9, "bold"), labelanchor="nw",
                               padx=12, pady=10, bd=1, relief="solid")
    info_frame.pack(fill="x", padx=14, pady=6)
    counters_text = lbl(info_frame, "Done: 0   Failed: 0   Up: 0m", font=("Segoe UI", 9))
    counters_text.pack(anchor="w")
    lbl(info_frame, f"Models: {', '.join(WORKER_MODELS)}",
        font=("Segoe UI", 9), fg=MUTED).pack(anchor="w", pady=(4, 0))
    lbl(info_frame, f"Worker ID: {WORKER_ID}   Capacity: {WORKER_CAPACITY}",
        font=("Segoe UI", 9), fg=MUTED).pack(anchor="w")

    # ── Buttons ───────────────────────────────────────────────────────
    btn_row = tk.Frame(root, bg=BG); btn_row.pack(fill="x", padx=14, pady=(10, 14))

    def open_dashboard(): webbrowser.open("https://genshape3d.com/dashboard")
    def open_log():
        log_path = REPO_ROOT / "logs" / "worker.log"
        if log_path.exists(): os.startfile(str(log_path))  # type: ignore[attr-defined]
    def open_folder(): os.startfile(str(REPO_ROOT))  # type: ignore[attr-defined]
    def quit_worker():
        with _active_lock:
            for jp in list(_active_jobs.values()):
                jp.cancel()
        root.destroy()
        os._exit(0)

    def mkbtn(text, cmd, color=BLUE):
        return tk.Button(btn_row, text=text, command=cmd, bg=color, fg="white",
                         activebackground=color, activeforeground="white",
                         relief="flat", bd=0, padx=12, pady=6,
                         font=("Segoe UI", 9), cursor="hand2")
    mkbtn("Dashboard", open_dashboard).pack(side="left", padx=(0, 6))
    mkbtn("Log file", open_log, "#21262d").pack(side="left", padx=6)
    mkbtn("Folder",   open_folder, "#21262d").pack(side="left", padx=6)
    mkbtn("Quit worker", quit_worker, RED).pack(side="right")

    # X-button minimizes instead of quits (matches 1080's Electron behaviour).
    root.protocol("WM_DELETE_WINDOW", root.iconify)

    # ── Refresh loop ──────────────────────────────────────────────────
    def refresh():
        try:
            # GPU stats
            if nvml_handle is not None:
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
                    temp = pynvml.nvmlDeviceGetTemperature(nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                    try: power = pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
                    except Exception: power = None
                    pct = (mem.used / mem.total) * 100 if mem.total else 0
                    vram_text.config(text=f"{_format_bytes(mem.used)} / {_format_bytes(mem.total)}  ({pct:.1f}%)")
                    vram_bar["value"] = pct
                    util_text.config(text=f"GPU util: {util.gpu}%")
                    temp_text.config(text=f"Temp: {temp}°C")
                    if power is not None:
                        power_text.config(text=f"Power: {power:.0f} W")
                except Exception:
                    pass

            # Worker state
            s = _state["status"]
            cj = _state["current_job"]
            if s == "working" and cj:
                elapsed = int(time.time() - cj["started"])
                status_text.config(text="Working")
                detail_text.config(text=f"Running {cj['model']} · job {cj['id'][:12]}… · {elapsed}s elapsed")
                status_dot.itemconfig(dot_id, fill=BLUE)
            elif s == "idle":
                status_text.config(text="Idle")
                detail_text.config(text="Waiting for jobs from server")
                status_dot.itemconfig(dot_id, fill=GREEN)
            elif s == "starting":
                status_text.config(text="Starting…")
                detail_text.config(text="Registering with server")
                status_dot.itemconfig(dot_id, fill=AMBER)
            else:
                status_text.config(text=s.capitalize())
                status_dot.itemconfig(dot_id, fill=AMBER)

            if cj:
                started_str = time.strftime("%H:%M:%S", time.localtime(cj["started"]))
                job_text.config(
                    text=f"Job ID: {cj['id']}\nModel:  {cj['model']}\nStarted: {started_str}",
                    fg=FG,
                )
            else:
                job_text.config(text="(none)", fg=MUTED)

            counters_text.config(text=(
                f"Done: {_state['jobs_done']}   "
                f"Failed: {_state['jobs_failed']}   "
                f"Up: {_format_uptime(time.time() - _state['started_at'])}"
            ))
        except Exception as e:
            print(f"[gui] refresh error: {e}", file=sys.stderr)
        root.after(1000, refresh)

    refresh()
    root.mainloop()

def main() -> None:
    if not WORKER_MODELS:
        print("WORKER_MODELS env var is empty", file=sys.stderr)
        sys.exit(1)
    _init_nvml()
    try:
        register()
    except Exception as e:
        print(f"[worker] register failed: {e}", file=sys.stderr)
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    threading.Thread(target=claim_loop_with_state, daemon=True).start()
    print(f"[worker] background threads started; serving UI on http://127.0.0.1:8765/")
    try:
        run_ui_server()
    except KeyboardInterrupt:
        print("[worker] interrupted; cancelling active jobs")
        with _active_lock:
            for jp in list(_active_jobs.values()):
                jp.cancel()

if __name__ == "__main__":
    main()
