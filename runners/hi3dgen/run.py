"""
Hi3DGen runner — image-to-3d via Stable-X/Hi3DGen.

Hi3DGen is a high-fidelity model that decomposes generation into normal-map
estimation followed by 3D reconstruction — slower than TripoSR/SF3D, often
better detail than vanilla Hunyuan3D.

NOTE: Hi3DGen's API is less stable than the others; verify on first run that
the import path + pipeline class names match the version cloned. Update if
the upstream repo has moved things around.

One-time bootstrap:
    git clone https://github.com/Stable-X/Hi3DGen.git C:\\projects\\ai\\hi3dgen\\Hi3DGen
    huggingface-cli download Stable-X/Hi3DGen \\
        --local-dir C:\\projects\\ai\\hi3dgen\\weights
"""

import argparse
import json
import os
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

# Where the Hi3DGen source repo lives (cloned during setup) and where weights live.
HI3DGEN_DIR = Path(os.environ.get("HI3DGEN_DIR", r"C:\projects\ai\hi3dgen\Hi3DGen"))
WEIGHTS_DIR = Path(os.environ.get("HI3DGEN_WEIGHTS", r"C:\projects\ai\hi3dgen\weights"))
sys.path.insert(0, str(HI3DGEN_DIR))

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

    fmt = (job.get("exportFormat") or "GLB").lower()
    if fmt not in ("glb", "obj", "ply", "stl"):
        fmt = "glb"
    out_file = out_dir / f"output.{fmt}"

    progress(2, "loading")

    try:
        import torch
        from PIL import Image
        # Hi3DGen's main pipeline class — adjust import if upstream renames.
        from hi3dgen.pipelines import Hi3DGenPipeline
    except ImportError as e:
        emit(type="failed", error=f"hi3dgen import failed: {e} (HI3DGEN_DIR={HI3DGEN_DIR})")
        return 1

    if not torch.cuda.is_available():
        emit(type="failed", error="CUDA not available")
        return 1

    progress(8, "loading")
    log(f"loading Hi3DGen from {WEIGHTS_DIR}")
    t0 = time.time()
    pipeline = Hi3DGenPipeline.from_pretrained(str(WEIGHTS_DIR))
    pipeline.cuda()
    log(f"pipeline loaded in {time.time() - t0:.1f}s")

    progress(25, "preprocessing")
    image = Image.open(image_path).convert("RGBA")

    seed = int(job.get("seed") or 0)
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed > 0 else None

    progress(35, "generating")
    t0 = time.time()
    with torch.no_grad():
        # Hi3DGen returns a trimesh-like object or a dict — adjust as needed.
        result = pipeline.run(image, generator=generator)
    log(f"generated in {time.time() - t0:.1f}s")

    # Some pipelines return {"mesh": <trimesh>} or a tuple — handle both.
    if hasattr(result, "export"):
        mesh = result
    elif isinstance(result, dict) and "mesh" in result:
        mesh = result["mesh"]
    elif isinstance(result, (list, tuple)) and result and hasattr(result[0], "export"):
        mesh = result[0]
    else:
        emit(type="failed", error=f"unexpected pipeline result type: {type(result)}")
        return 1

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
