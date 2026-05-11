"""
Hi3DGen runner (project renamed to Stable3DGen).
Uses Stable-X/trellis-normal-v0-1 weights via Stable3DGen's Hi3DGenPipeline.

CLI + stdout protocol matches genshape3d_nvidia/generate.py so the
Electron worker.js can spawn this just like it spawns generate.py.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

STABLE3DGEN_DIR = Path(os.environ.get("STABLE3DGEN_DIR", r"C:\projects\ai\hi3dgen\Stable3DGen"))
sys.path.insert(0, str(STABLE3DGEN_DIR))

WEIGHTS_DIR = Path(r"C:\projects\ai\hi3dgen\trellis-normal-v0-1")


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
        from hi3dgen.pipelines import Hi3DGenPipeline
    except ImportError as e:
        print(f"RESULT:{json.dumps({'status':'error','error':f'import: {e}'})}", flush=True)
        return 1

    if not torch.cuda.is_available():
        print(f"RESULT:{json.dumps({'status':'error','error':'CUDA not available'})}", flush=True)
        return 1

    emit_progress(5, "loading", detail="Loading Hi3DGen...")
    t0 = time.time()
    pipeline = Hi3DGenPipeline.from_pretrained(str(WEIGHTS_DIR))
    pipeline.cuda()
    print(f"[hi3dgen] pipeline loaded in {time.time() - t0:.1f}s", flush=True)
    emit_progress(25, "analyzing", detail="Analyzing image...")

    image = Image.open(args.image).convert("RGBA")
    generator = torch.Generator(device="cuda").manual_seed(args.seed) if args.seed > 0 else None

    emit_progress(35, "generating", step=1, total=1, detail="Generating geometry...")
    t1 = time.time()
    with torch.no_grad():
        result = pipeline.run(image, generator=generator)
    print(f"[hi3dgen] generated in {time.time() - t1:.1f}s", flush=True)

    # Stable3DGen's pipeline.run returns a trimesh-like object or a dict.
    if hasattr(result, "export"):
        mesh = result
    elif isinstance(result, dict) and "mesh" in result:
        mesh = result["mesh"]
    elif isinstance(result, (list, tuple)) and result and hasattr(result[0], "export"):
        mesh = result[0]
    else:
        print(f"RESULT:{json.dumps({'status':'error','error':f'unexpected pipeline result type: {type(result).__name__}'})}", flush=True)
        return 1

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
