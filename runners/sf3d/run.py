"""
Stable Fast 3D (SF3D) runner — image-to-3d via stabilityai/stable-fast-3d.

Honours the worker→runner contract from ../../SETUP.md.

SF3D is fast (~1-2s on a 3090 after model load) and produces UV-mapped
textures natively. Different code path from Hunyuan3D / TripoSR.

One-time bootstrap:
    huggingface-cli download stabilityai/stable-fast-3d \\
        config.yaml model.safetensors \\
        --local-dir C:\\projects\\ai\\sf3d
"""

import argparse
import json
import sys
import time
from pathlib import Path

def emit(**kw) -> None:
    sys.stdout.write(json.dumps(kw) + "\n")
    sys.stdout.flush()

def progress(pct: int, phase: str, step: int | None = None, total: int | None = None) -> None:
    payload = {"type": "progress", "pct": int(pct), "phase": phase}
    if step is not None: payload["step"] = step
    if total is not None: payload["total"] = total
    emit(**payload)

def log(msg: str) -> None:
    emit(type="log", msg=msg)

WEIGHTS_DIR = Path(r"C:\projects\ai\sf3d")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-json", required=True)
    ap.add_argument("--image-path", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
    image_path = Path(args.image_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "output.glb"

    progress(2, "loading")

    try:
        import torch
        from PIL import Image
        from sf3d.system import SF3D
    except ImportError as e:
        emit(type="failed", error=f"sf3d import failed: {e}")
        return 1

    if not torch.cuda.is_available():
        emit(type="failed", error="CUDA not available")
        return 1

    progress(8, "loading")
    log(f"loading SF3D from {WEIGHTS_DIR}")
    t0 = time.time()
    model = SF3D.from_pretrained(
        str(WEIGHTS_DIR),
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.eval().cuda()
    log(f"model loaded in {time.time() - t0:.1f}s")

    progress(25, "preprocessing")
    image = Image.open(image_path).convert("RGBA")

    # Map our textureRes string to SF3D's bake_resolution int.
    tex_res = (job.get("textureRes") or "1K").upper()
    bake_res = {"1K": 1024, "2K": 2048, "4K": 4096}.get(tex_res, 1024)
    log(f"bake_resolution={bake_res}")

    progress(40, "generating")
    t0 = time.time()
    with torch.no_grad():
        mesh, _texture, _glb_buf = model.run_image(
            image,
            bake_resolution=bake_res,
            remesh="none",
            vertex_count=-1,
            return_points=False,
        )
    log(f"generated in {time.time() - t0:.1f}s")

    progress(92, "exporting")
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
