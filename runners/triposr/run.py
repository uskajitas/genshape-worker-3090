"""
TripoSR runner — image-to-3d via stabilityai/TripoSR.

Honours the worker→runner contract from ../../SETUP.md: reads CLI args, emits
JSON Lines on stdout, writes the final mesh to --output-dir.

Weights are pre-downloaded to C:\\projects\\ai\\triposr\\ (see WEIGHTS_DIR
below). To bootstrap once:
    huggingface-cli download stabilityai/TripoSR \\
        config.yaml model.ckpt \\
        --local-dir C:\\projects\\ai\\triposr
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# TripoSR is not a pip package — its source repo must be cloned and added
# to sys.path. Default clone target sits next to the weights so both can be
# moved or backed up together.
TRIPOSR_DIR = Path(os.environ.get("TRIPOSR_DIR", r"C:\projects\ai\triposr\TripoSR"))
sys.path.insert(0, str(TRIPOSR_DIR))

# ─── JSONL event emitter ─────────────────────────────────────────────────────
def emit(**kw) -> None:
    sys.stdout.write(json.dumps(kw) + "\n")
    sys.stdout.flush()

def progress(pct: int, phase: str, step: int | None = None, total: int | None = None) -> None:
    payload = {"type": "progress", "pct": pct, "phase": phase}
    if step is not None: payload["step"] = step
    if total is not None: payload["total"] = total
    emit(**payload)

def log(msg: str) -> None:
    emit(type="log", msg=msg)

WEIGHTS_DIR = Path(r"C:\projects\ai\triposr")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-json", required=True)
    ap.add_argument("--image-path", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
    image_path = Path(args.image_path)
    out_dir = Path(args.output_dir)

    export_format = (job.get("exportFormat") or "GLB").lower()
    if export_format not in ("glb", "obj"):
        # TripoSR's mesh export covers GLB and OBJ comfortably; PLY/STL fall back to GLB.
        export_format = "glb"
    out_file = out_dir / f"output.{export_format}"

    progress(2, "loading")

    # Imports are inside main() so a missing dep doesn't crash the worker
    # before we get a chance to emit a structured failure event.
    try:
        import torch
        from PIL import Image
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground
        import rembg
    except ImportError as e:
        emit(type="failed", error=f"runner import failed: {e}")
        return 1

    if not torch.cuda.is_available():
        emit(type="failed", error="CUDA not available")
        return 1

    progress(8, "loading")
    log(f"loading TripoSR from {WEIGHTS_DIR}")
    t0 = time.time()
    model = TSR.from_pretrained(
        str(WEIGHTS_DIR),
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to("cuda")
    log(f"model loaded in {time.time() - t0:.1f}s")

    progress(20, "preprocessing")
    image = Image.open(image_path).convert("RGBA")
    # Server already strips backgrounds before queueing, but TripoSR ships its
    # own resize_foreground step that crops/centres the subject for the
    # network. If the alpha channel is non-trivial we use it directly; else
    # we run a quick rembg pass as a safety net.
    if image.getextrema()[3] != (255, 255):
        log("input already has alpha; skipping rembg")
    else:
        log("input has no alpha; running rembg")
        session = rembg.new_session()
        image = remove_background(image, session)
    image = resize_foreground(image, 0.85)

    progress(35, "generating")
    t0 = time.time()
    with torch.no_grad():
        scene_codes = model([image], device="cuda")
    log(f"scene codes in {time.time() - t0:.1f}s")

    progress(70, "extracting")
    t0 = time.time()
    # TripoSR's extract_mesh resolution: 256 default, 384/512 = nicer + slower.
    requested_octree = int(job.get("octreeResolution") or 0)
    mesh_res = requested_octree if requested_octree in (256, 320, 384, 512) else 256
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mesh_res)
    mesh = meshes[0]
    log(f"mesh extracted ({len(mesh.vertices)} verts) in {time.time() - t0:.1f}s")

    progress(92, "exporting")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_file))
    log(f"wrote {out_file}")

    emit(type="done", filename=out_file.name)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit(type="failed", error=f"{type(e).__name__}: {e}")
        sys.exit(1)
