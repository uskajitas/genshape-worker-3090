"""
One-shot script to pre-download model weights for the 3090 worker.

Reads HF_TOKEN from .env, downloads each model's weights to its
C:\\projects\\ai\\<model>\\ directory. Idempotent — re-running just
re-checks file hashes and skips already-downloaded files.

Hunyuan3D is intentionally skipped here: the runner uses
Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2",
subfolder=...) which auto-downloads to the HF cache on first job.

Usage:
    cd C:\\projects\\genshape-worker-3090
    .\\.venv\\Scripts\\python.exe bootstrap-weights.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

token = os.environ.get("HF_TOKEN")
if not token:
    print("HF_TOKEN not set in .env", file=sys.stderr)
    sys.exit(1)

# (HF repo id, local dir, optional list of allow patterns)
JOBS = [
    ("stabilityai/TripoSR",           r"C:\projects\ai\triposr",  ["config.yaml", "model.ckpt"]),
    ("stabilityai/stable-fast-3d",    r"C:\projects\ai\sf3d",     None),  # full repo — not huge
    ("Stable-X/trellis-normal-v0-1",  r"C:\projects\ai\hi3dgen\trellis-normal-v0-1", None),
    ("Stable-X/yoso-normal-v1-8-1",   r"C:\projects\ai\hi3dgen\yoso-normal-v1-8-1",  None),
]

for repo_id, local_dir, allow_patterns in JOBS:
    print(f"\n=== {repo_id} -> {local_dir} ===", flush=True)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            token=token,
        )
        size = sum(f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file())
        print(f"  OK in {time.time() - t0:.1f}s ({size / 1e9:.2f} GB on disk)", flush=True)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

print("\nDone.")
