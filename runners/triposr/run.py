"""
TripoSR runner — image-to-3d via stabilityai/TripoSR.

CLI + stdout protocol intentionally match genshape3d_nvidia's generate.py:
  --image / --output / --steps / etc.
  stdout: PROGRESS:{...json...} lines + final RESULT:{...} line.
That way the 1080's Electron worker.js spawns this script the same way
it spawns generate.py — just with a different python + path.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

TRIPOSR_DIR = Path(os.environ.get("TRIPOSR_DIR", r"C:\projects\ai\triposr\TripoSR"))
sys.path.insert(0, str(TRIPOSR_DIR))

WEIGHTS_DIR = Path(r"C:\projects\ai\triposr")


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
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground
    except ImportError as e:
        print(f"RESULT:{json.dumps({'status':'error','error':f'import: {e}'})}", flush=True)
        return 1

    if not torch.cuda.is_available():
        print(f"RESULT:{json.dumps({'status':'error','error':'CUDA not available'})}", flush=True)
        return 1

    emit_progress(5, "loading", detail="Loading TripoSR...")
    t0 = time.time()
    model = TSR.from_pretrained(str(WEIGHTS_DIR), config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(args.num_chunks if args.num_chunks else 8192)
    model.to("cuda")
    print(f"[triposr] model loaded in {time.time() - t0:.1f}s", flush=True)
    emit_progress(15, "analyzing", detail="Analyzing image...")

    image = Image.open(args.image).convert("RGBA")
    if args.remove_bg and image.getextrema()[3] == (255, 255):
        import rembg
        session = rembg.new_session()
        image = remove_background(image, session)
    image = resize_foreground(image, 0.85)

    emit_progress(30, "generating", step=1, total=2, detail="Generating geometry...")
    t1 = time.time()
    with torch.no_grad():
        scene_codes = model([image], device="cuda")

    emit_progress(70, "generating", step=2, total=2, detail="Extracting mesh...")
    mesh_res = args.octree_resolution if args.octree_resolution in (256, 320, 384, 512) else 256
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mesh_res)
    mesh = meshes[0]
    print(f"[triposr] mesh extracted ({len(mesh.vertices)} verts) in {time.time() - t1:.1f}s", flush=True)

    emit_progress(90, "refining", detail="Refining mesh...")
    if args.target_face_count and len(mesh.faces) > args.target_face_count:
        try:
            import pymeshlab as pml
            ms = pml.MeshSet()
            ms.add_mesh(pml.Mesh(mesh.vertices, mesh.faces))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=args.target_face_count)
            simp = ms.current_mesh()
            import trimesh as tm
            mesh = tm.Trimesh(vertices=simp.vertex_matrix(), faces=simp.face_matrix())
        except Exception as e:
            print(f"[triposr] face reduction skipped: {e}", flush=True)

    emit_progress(95, "exporting", detail="Exporting...")
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
