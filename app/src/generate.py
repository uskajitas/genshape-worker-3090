"""
Hunyuan3D direct inference script.
Called by the Electron worker as a subprocess.

Usage:
  python generate.py --image input.png --output output.glb [options]

Outputs PROGRESS: lines to stdout for the Electron worker to parse.
Each PROGRESS line contains the OVERALL job progress (0-100%) with clear phase info.

Phases and their weight in overall progress:
  - loading     (0-5%)    : Load model into GPU
  - diffusion   (5-15%)   : Diffusion sampling (fast, 5-30 steps)
  - decoding    (15-85%)  : Volume decoding (slow, the bulk of the work)
  - simplifying (85-90%)  : Mesh face reduction
  - exporting   (90-95%)  : Export to GLB/OBJ
  - texture     (varies)  : Optional texture generation

Exits with code 0 on success, 1 on error.
"""

import argparse
import sys
import os
import json
import time
import re

HUNYUAN_DIR = os.environ.get('HUNYUAN3D_DIR', 'F:/ai/hunyuan3d-2')
sys.path.insert(0, HUNYUAN_DIR)


def emit_progress(pct, phase, step=0, total=0, detail=''):
    """Print a PROGRESS line for the Electron worker."""
    obj = {
        'pct': min(pct, 100),
        'phase': phase,
        'step': step,
        'total': total,
        'detail': detail,
    }
    print(f'PROGRESS:{json.dumps(obj)}', flush=True)


class TqdmProgressHook:
    """
    Monkey-patches tqdm to capture volume decoding progress
    and emit PROGRESS lines with correct overall percentages.

    Volume decoding maps to 15-85% of overall progress.
    """
    def __init__(self):
        self.original_tqdm = None
        self.active = False

    def install(self):
        import tqdm as tqdm_module
        self.original_tqdm = tqdm_module.tqdm

        parent = self

        class HookedTqdm(self.original_tqdm):
            def __init__(self, iterable=None, desc=None, total=None, **kwargs):
                super().__init__(iterable, desc=desc, total=total, **kwargs)
                self._hook_desc = desc or ''
                self._hook_total = total or (len(iterable) if iterable is not None and hasattr(iterable, '__len__') else 0)

            def update(self, n=1):
                super().update(n)
                if 'Volume Decoding' in self._hook_desc and self._hook_total > 0:
                    vol_pct = self.n / self._hook_total
                    # Volume decoding maps to 15% - 85% of overall progress
                    overall_pct = round(15 + vol_pct * 70)
                    emit_progress(
                        overall_pct,
                        'generating',
                        step=self.n,
                        total=self._hook_total,
                        detail=f'Building 3D structure... {overall_pct}%'
                    )
                elif 'Diffusion' in self._hook_desc and self._hook_total > 0:
                    diff_pct = self.n / self._hook_total
                    # Diffusion maps to 5% - 15% of overall progress
                    overall_pct = round(5 + diff_pct * 10)
                    emit_progress(
                        overall_pct,
                        'analyzing',
                        step=self.n,
                        total=self._hook_total,
                        detail=f'Analyzing image... {overall_pct}%'
                    )

        tqdm_module.tqdm = HookedTqdm
        # Also patch the submodule that shapegen imports from
        try:
            import hy3dgen.shapegen.models.autoencoders.volume_decoders as vd
            vd.tqdm = HookedTqdm
        except:
            pass
        try:
            import hy3dgen.shapegen.pipelines as pl
            pl.tqdm = HookedTqdm
        except:
            pass

    def uninstall(self):
        if self.original_tqdm:
            import tqdm as tqdm_module
            tqdm_module.tqdm = self.original_tqdm


def main():
    parser = argparse.ArgumentParser(description='Hunyuan3D mesh generation')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output mesh file path')
    parser.add_argument('--steps', type=int, default=5, help='Inference steps (5=turbo, 15=fast, 30=standard)')
    parser.add_argument('--guidance-scale', type=float, default=5.0, help='Guidance scale')
    parser.add_argument('--octree-resolution', type=int, default=384, help='Octree resolution (256/384/512)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (0=random)')
    parser.add_argument('--num-chunks', type=int, default=200000, help='Num chunks for volume decoder')
    parser.add_argument('--do-texture', action='store_true', help='Generate texture')
    parser.add_argument('--target-face-count', type=int, default=100000, help='Target face count')
    parser.add_argument('--export-format', type=str, default='glb', help='Export format: glb, obj, ply, stl')
    parser.add_argument('--remove-bg', action='store_true', default=True, help='Remove background')
    args = parser.parse_args()

    print(f'[generate.py] Loading image: {args.image}', flush=True)
    print(f'[generate.py] Params: steps={args.steps}, guidance={args.guidance_scale}, '
          f'octree={args.octree_resolution}, faces={args.target_face_count}, '
          f'texture={args.do_texture}, format={args.export_format}', flush=True)

    t0 = time.time()

    emit_progress(0, 'starting', detail='Preparing...')

    import torch
    from PIL import Image

    # Load image
    image = Image.open(args.image)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Remove background
    if args.remove_bg:
        try:
            from hy3dgen.shapegen import rembg
            emit_progress(1, 'starting', detail='Processing image...')
            bg_remover = rembg.BackgroundRemover()
            image = bg_remover(image)
        except Exception as e:
            print(f'[generate.py] Background removal failed (continuing): {e}', flush=True)

    # Install tqdm hook BEFORE importing the pipeline
    hook = TqdmProgressHook()
    hook.install()

    # Load shape generation pipeline
    emit_progress(2, 'loading', detail='Loading AI model...')
    print('[generate.py] Loading shape generation pipeline...', flush=True)
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if args.steps <= 10:
        subfolder = 'hunyuan3d-dit-v2-0-turbo'
    else:
        subfolder = 'hunyuan3d-dit-v2-0'

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        subfolder=subfolder,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=True,
    )

    emit_progress(5, 'loading', detail='Model ready, starting generation...')

    # Set up generator for seed
    generator = None
    if args.seed > 0:
        generator = torch.Generator(device='cuda').manual_seed(args.seed)

    # Generate mesh
    # The tqdm hook will emit progress for diffusion (5-15%) and volume decoding (15-85%)
    print(f'[generate.py] Generating 3D mesh ({args.steps} steps)...', flush=True)
    emit_progress(5, 'analyzing', step=0, total=args.steps, detail='Analyzing image...')

    outputs = pipeline(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        output_type='mesh',
    )

    # Convert Latent2MeshOutput to trimesh
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    mesh = export_to_trimesh(outputs)[0]

    t1 = time.time()
    emit_progress(85, 'refining', detail='Refining 3D model...')
    print(f'[generate.py] Shape generation done in {t1 - t0:.1f}s', flush=True)
    print(f'[generate.py] Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces', flush=True)

    # Simplify mesh
    if args.target_face_count > 0 and len(mesh.faces) > args.target_face_count:
        try:
            from hy3dgen.shapegen.postprocessors import FaceReducer
            emit_progress(87, 'refining', detail='Optimizing mesh...')
            reducer = FaceReducer()
            mesh = reducer(mesh, args.target_face_count)
            print(f'[generate.py] After simplification: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces', flush=True)
        except Exception as e:
            print(f'[generate.py] Face reduction failed (keeping original): {e}', flush=True)

    # Texture
    if args.do_texture:
        try:
            emit_progress(90, 'texturing', detail='Applying textures...')
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
            mesh = tex_pipeline(mesh, image=image)
            print('[generate.py] Texture generation done', flush=True)
        except Exception as e:
            print(f'[generate.py] Texture generation failed (exporting without): {e}', flush=True)

    # Export
    emit_progress(93, 'exporting', detail='Exporting 3D file...')
    output_path = args.output
    if not output_path.endswith(f'.{args.export_format}'):
        output_path = os.path.splitext(output_path)[0] + f'.{args.export_format}'

    mesh.export(output_path)

    t2 = time.time()
    file_size = os.path.getsize(output_path)
    emit_progress(100, 'done', detail='Generation complete!')
    print(f'[generate.py] Done! Total time: {t2 - t0:.1f}s, file size: {file_size} bytes', flush=True)

    hook.uninstall()

    result = {
        'status': 'success',
        'output_path': output_path,
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'file_size': file_size,
        'generation_time': round(t1 - t0, 1),
        'total_time': round(t2 - t0, 1),
    }
    print(f'RESULT:{json.dumps(result)}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'[generate.py] FATAL ERROR: {e}', file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        error_result = {'status': 'error', 'error': str(e)}
        print(f'RESULT:{json.dumps(error_result)}', flush=True)
        sys.exit(1)
