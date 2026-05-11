"""
Hunyuan3D-2 runner — image-to-3d via tencent/Hunyuan3D-2.

Ported from genshape3d_nvidia/src/generate.py (the i7 worker's Python script).
Keeps the same tqdm-monkeypatch trick to surface fine-grained progress for
the slow Volume Decoding phase, but adapted to the new worker→runner JSONL
contract (no more PROGRESS:{...} / RESULT:{...} lines).

Weights live at C:\\projects\\ai\\hunyuan3d\\ — point HUNYUAN3D_DIR there
(see env var below) so the hy3dgen package can be imported and the
checkpoints loaded.

One-time bootstrap for this runner's venv:
    py -3.11 -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    pip install --index-url https://download.pytorch.org/whl/cu121 \\
        torch==2.3.0 torchvision==0.18.0
    pip install -r requirements.txt
    # Clone hy3dgen (the model code) into HUNYUAN3D_DIR; weights download
    # automatically on first run via huggingface_hub when CHECKPOINT cache
    # is empty.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ─── JSONL event emitter (the worker→runner contract) ───────────────────────
def emit(**kw) -> None:
    sys.stdout.write(json.dumps(kw) + "\n")
    sys.stdout.flush()

def progress(pct: int, phase: str, step: int | None = None, total: int | None = None) -> None:
    payload = {"type": "progress", "pct": min(int(pct), 100), "phase": phase}
    if step is not None: payload["step"] = step
    if total is not None: payload["total"] = total
    emit(**payload)

def log(msg: str) -> None:
    emit(type="log", msg=msg)

# Local copy of the hy3dgen package — point HUNYUAN3D_DIR at where the repo
# is cloned. Defaults to a sibling of the model weights.
HUNYUAN_DIR = os.environ.get("HUNYUAN3D_DIR", r"C:\projects\ai\hunyuan3d\Hunyuan3D-2")
sys.path.insert(0, HUNYUAN_DIR)


# ─── Progress hook ───────────────────────────────────────────────────────────
class TqdmProgressHook:
    """
    Monkey-patches tqdm so volume-decoding + diffusion progress maps onto our
    overall percentage scale (5-15% diffusion, 15-85% volume decoding).
    Without this, the user sees a 60-second silent gap on big jobs.
    """
    def __init__(self):
        self.original_tqdm = None

    def install(self):
        import tqdm as tqdm_module
        self.original_tqdm = tqdm_module.tqdm

        class HookedTqdm(self.original_tqdm):
            def __init__(self, iterable=None, desc=None, total=None, **kwargs):
                super().__init__(iterable, desc=desc, total=total, **kwargs)
                self._hook_desc = desc or ""
                self._hook_total = total or (
                    len(iterable) if iterable is not None and hasattr(iterable, "__len__") else 0
                )

            def update(self, n=1):
                super().update(n)
                if "Volume Decoding" in self._hook_desc and self._hook_total > 0:
                    overall = round(15 + (self.n / self._hook_total) * 70)
                    progress(overall, "generating", step=self.n, total=self._hook_total)
                elif "Diffusion" in self._hook_desc and self._hook_total > 0:
                    overall = round(5 + (self.n / self._hook_total) * 10)
                    progress(overall, "analyzing", step=self.n, total=self._hook_total)

        tqdm_module.tqdm = HookedTqdm
        # Also patch the submodules that already imported tqdm at module load time.
        for modname in (
            "hy3dgen.shapegen.models.autoencoders.volume_decoders",
            "hy3dgen.shapegen.pipelines",
        ):
            try:
                __import__(modname)
                sys.modules[modname].tqdm = HookedTqdm
            except Exception:
                pass

    def uninstall(self):
        if self.original_tqdm:
            import tqdm as tqdm_module
            tqdm_module.tqdm = self.original_tqdm


# ─── Main ────────────────────────────────────────────────────────────────────
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

    # Normalise params (server may leave 0 = default).
    steps        = int(job.get("inferenceSteps") or 5)
    guidance     = float(job.get("guidanceScale") or 5.0)
    octree_res   = int(job.get("octreeResolution") or 384)
    num_chunks   = int(job.get("numChunks") or 200000)
    target_faces = int(job.get("targetFaceCount") or 100000)
    seed         = int(job.get("seed") or 0)
    do_texture   = bool(job.get("doTexture"))
    fmt          = (job.get("exportFormat") or "GLB").lower()
    if fmt not in ("glb", "obj", "ply", "stl"):
        fmt = "glb"
    out_file = out_dir / f"output.{fmt}"

    progress(1, "starting")
    log(f"hunyuan3d steps={steps} guidance={guidance} octree={octree_res} "
        f"faces={target_faces} texture={do_texture} format={fmt}")

    try:
        import torch
        from PIL import Image
    except ImportError as e:
        emit(type="failed", error=f"torch/PIL import failed: {e}")
        return 1

    if not torch.cuda.is_available():
        emit(type="failed", error="CUDA not available")
        return 1

    # Load image
    image = Image.open(image_path)
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Background removal — server already strips, but keep this as safety net.
    try:
        from hy3dgen.shapegen import rembg
        progress(2, "starting")
        bg_remover = rembg.BackgroundRemover()
        image = bg_remover(image)
    except Exception as e:
        log(f"rembg fallback skipped: {e}")

    hook = TqdmProgressHook()
    hook.install()

    progress(3, "loading")
    log("loading Hunyuan3D pipeline")
    t0 = time.time()
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    subfolder = "hunyuan3d-dit-v2-0-turbo" if steps <= 10 else "hunyuan3d-dit-v2-0"
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder=subfolder,
        device="cuda",
        dtype=torch.float16,
        use_safetensors=True,
    )
    log(f"pipeline loaded ({subfolder}) in {time.time() - t0:.1f}s")
    progress(5, "loading")

    generator = torch.Generator(device="cuda").manual_seed(seed) if seed > 0 else None

    progress(5, "analyzing", step=0, total=steps)
    t1 = time.time()
    outputs = pipeline(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        octree_resolution=octree_res,
        num_chunks=num_chunks,
        output_type="mesh",
    )

    from hy3dgen.shapegen.pipelines import export_to_trimesh
    mesh = export_to_trimesh(outputs)[0]
    log(f"shape generated ({len(mesh.vertices)} verts, {len(mesh.faces)} faces) in {time.time() - t1:.1f}s")
    progress(85, "refining")

    # Face reduction
    if target_faces > 0 and len(mesh.faces) > target_faces:
        try:
            from hy3dgen.shapegen.postprocessors import FaceReducer
            progress(87, "refining")
            reducer = FaceReducer()
            mesh = reducer(mesh, target_faces)
            log(f"face-reduced to {len(mesh.faces)} faces")
        except Exception as e:
            log(f"face reduction skipped: {e}")

    # Texture
    if do_texture:
        try:
            progress(90, "texturing")
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
            mesh = tex_pipeline(mesh, image=image)
            log("texture generated")
        except Exception as e:
            log(f"texture skipped: {e}")

    progress(95, "exporting")
    mesh.export(str(out_file))
    hook.uninstall()
    log(f"exported {out_file}")

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
