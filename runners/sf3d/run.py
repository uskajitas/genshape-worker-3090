"""
Stable Fast 3D runner — image-to-3d via stabilityai/stable-fast-3d.

CLI + stdout protocol matches genshape3d_nvidia/generate.py so the 1080's
Electron worker.js can spawn this just like it spawns generate.py.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SF3D_REPO_DIR = Path(os.environ.get("SF3D_REPO_DIR", r"C:\projects\ai\sf3d\stable-fast-3d"))
sys.path.insert(0, str(SF3D_REPO_DIR))

WEIGHTS_DIR = Path(r"C:\projects\ai\sf3d")


def emit_progress(pct, phase, step=0, total=0, detail=""):
    obj = {"pct": min(int(pct), 100), "phase": phase, "step": int(step), "total": int(total), "detail": detail}
    print(f"PROGRESS:{json.dumps(obj)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--octree-resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=8000)
    ap.add_argument("--target-face-count", type=int, default=30000)
    ap.add_argument("--export-format", default="glb")
    ap.add_argument("--remove-bg", action="store_true")
    ap.add_argument("--do-texture", action="store_true")
    args = ap.parse_args()

    t0_total = time.time()
    emit_progress(0, "starting", detail="Preparing...")

    try:
        import torch
        from PIL import Image
        from sf3d.system import SF3D
    except ImportError as e:
        print(f"RESULT:{json.dumps({'status':'error','error':f'import: {e}'})}", flush=True)
        return 1

    if not torch.cuda.is_available():
        print(f"RESULT:{json.dumps({'status':'error','error':'CUDA not available'})}", flush=True)
        return 1

    emit_progress(5, "loading", detail="Loading SF3D...")
    t0 = time.time()
    model = SF3D.from_pretrained(str(WEIGHTS_DIR), config_name="config.yaml", weight_name="model.safetensors")
    model.eval().cuda()
    print(f"[sf3d] model loaded in {time.time() - t0:.1f}s", flush=True)
    emit_progress(25, "analyzing", detail="Analyzing image...")

    image = Image.open(args.image).convert("RGBA")

    # Map texture-res hint from job: detail level or explicit.
    bake_res = 1024  # 1K default — fast. Worker may override later.

    emit_progress(40, "generating", step=1, total=1, detail="Generating mesh + texture...")
    t1 = time.time()
    with torch.no_grad():
        mesh, _texture, _glb_buf = model.run_image(
            image,
            bake_resolution=bake_res,
            remesh="none",
            vertex_count=-1,
            return_points=False,
        )
    print(f"[sf3d] generated in {time.time() - t1:.1f}s", flush=True)

    emit_progress(92, "exporting", detail="Exporting...")
    out_path = args.output
    if not out_path.lower().endswith(f".{args.export_format.lower()}"):
        out_path = os.path.splitext(out_path)[0] + f".{args.export_format.lower()}"
    mesh.export(out_path)
    size = os.path.getsize(out_path)
    emit_progress(100, "done", detail="Generation complete!")

    print(f"RESULT:{json.dumps({'status':'success','output_path':out_path,'vertices':len(mesh.vertices),'faces':len(mesh.faces),'file_size':size,'total_time':round(time.time()-t0_total,1)})}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"RESULT:{json.dumps({'status':'error','error':f'{type(e).__name__}: {e}'})}", flush=True)
        sys.exit(1)
