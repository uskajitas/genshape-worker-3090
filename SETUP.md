# genshape-worker-3090

Multi-model GPU worker for the genshape3d image-to-3D pipeline. Runs on the
RTX 3090 home box. Registers with the API server, long-polls for jobs whose
declared `model` matches one this worker advertises, and dispatches each job
to the matching runner subprocess.

## Repo layout

```
genshape-worker-3090/
├── worker.py              # The dispatcher. Long-lived, no ML deps.
├── requirements.txt       # Worker-shell deps only (requests, boto3, dotenv).
├── .env.example           # Copy to .env and fill in.
├── .venv/                 # Worker-shell venv (created by setup, gitignored).
└── runners/
    ├── hunyuan3d/
    │   ├── run.py         # Entrypoint, loads + runs Hunyuan3D-2.
    │   ├── requirements.txt
    │   └── .venv/         # Per-model venv (gitignored).
    ├── triposr/
    │   ├── run.py
    │   ├── requirements.txt
    │   └── .venv/
    ├── sf3d/
    │   ├── run.py
    │   ├── requirements.txt
    │   └── .venv/
    └── hi3dgen/
        ├── run.py
        ├── requirements.txt
        └── .venv/
```

Model **weights** live outside the repo at `C:\projects\ai\<model>\` so they
don't get accidentally copied around with the code.

## Why per-runner venvs

Each model has incompatible dependencies — Hunyuan3D wants `hy3dgen`, SF3D
wants `gsplat`, TripoSR uses an older diffusers, Hi3DGen has its own
checkpoints loader. Forcing them into one venv guarantees breakage. Separate
venvs cost disk (~3-4 GB each from the torch wheel) but eliminate the entire
class of "model A broke after I installed model B" failures.

## Worker→runner contract

The dispatcher spawns each runner as a subprocess:

```
<runners/<model>/.venv/Scripts/python.exe>  runners/<model>/run.py \
    --job-json <path>     # full job payload as JSON file
    --image-path <path>   # local PNG already downloaded
    --output-dir <path>   # write the result mesh here
```

Runner emits **JSON Lines** on stdout, one event per line:

```
{"type":"progress","pct":12,"phase":"loading"}
{"type":"progress","pct":50,"phase":"generating","step":15,"total":30}
{"type":"log","msg":"loaded checkpoint"}
{"type":"done","filename":"output.glb"}
```

Or on failure:
```
{"type":"failed","error":"OOM during decoding"}
```

Then the runner exits. Exit code 0 = success (must be preceded by a `done`
event), non-zero = failure. Stderr is for tracebacks / unstructured logs and
is forwarded to the worker's stderr but not parsed.

The worker uploads `output-dir / <filename>` to R2 after a successful run
and reports the URL back to the server via `/api/workers/<id>/complete`.

## First-time setup

### 1. Worker shell
```powershell
cd C:\projects\genshape-worker-3090
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env: paste WORKER_AUTH_TOKEN and R2 creds (from i7 password manager).
```

### 2. Per-runner setup

Each runner has its own venv with its own torch wheel. Same recipe applies
to all four; only the model-specific deps differ.

#### Step 1: build environment
The MSVC v14.44 toolset that ships with VS BuildTools 17.10+ is **rejected
by CUDA 12.1** (yvals_core.h has a `static_assert` requiring CUDA ≥ 12.4).
Install the older v14.39 toolset side-by-side:
1. Open **Visual Studio Installer** → **Modify** Build Tools 2022.
2. **Individual components** tab → search `v14.39`.
3. Tick **MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.39-17.9)**.
4. Modify, wait ~10 min.

Always run `pip install` for runner deps from a developer command prompt
with the v14.39 toolset selected:
```cmd
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39
```

CUDA 12.1's "Visual Studio Integration" sub-component must also be
installed (or its 4 files manually copied to
`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations\`).
Without it CMake errors with "No CUDA toolset found."

#### Step 2: create venv + install deps
```powershell
cd C:\projects\genshape-worker-3090\runners\<model>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# ⚠ Pin torch 2.5.1+cu121 (NOT older — see Step 3 for why).
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
```

#### Step 3: NVTX3 patch (REQUIRED for any C++/CUDA extension build)
PyTorch's `cuda.cmake` looks for NVTX3 headers at a path that's empty in
the wheel. If found, it uses header-only NVTX3 — perfect. If not, it
falls back to the legacy `CUDA::nvToolsExt` link target which doesn't
exist on CUDA 12.x. Every `pip install` of a package with a CUDA
extension (torchmcubes, gsplat, custom torch ops, etc.) hits this.

We bootstrapped the headers once at
`C:\projects\ai\nvtx_redist\extracted\nvidia\nvtx\include\` (extracted
from the `nvidia-nvtx-cu12` PyPI wheel). Patch each runner's torch CMake
config to look there too:

Edit `runners\<model>\.venv\Lib\site-packages\torch\share\cmake\Caffe2\public\cuda.cmake`,
find this line:
```cmake
find_path(nvtx3_dir NAMES nvtx3 PATHS "${PROJECT_SOURCE_DIR}/third_party/NVTX/c/include" NO_DEFAULT_PATH)
```
and change it to:
```cmake
find_path(nvtx3_dir NAMES nvtx3 PATHS "${PROJECT_SOURCE_DIR}/third_party/NVTX/c/include" "C:/projects/ai/nvtx_redist/extracted/nvidia/nvtx/include" NO_DEFAULT_PATH)
```

The patch must be re-applied if torch is reinstalled (it lives inside the
torch wheel).

#### Step 4: install runner-specific deps
```powershell
pip install -r requirements.txt
```

#### Step 4a: Hi3DGen-specific gotchas (Stable3DGen)
Hi3DGen's repo was renamed to **Stable3DGen**; the importable Python
package is still called `hi3dgen`. Two differences from the other runners:

1. **torch is pinned to 2.4.0+cu121, NOT 2.5.1.** Reason: `xformers==0.0.27.post2`
   (an upstream pin) is built against torch 2.4. Trying to install it on
   torch 2.5 fails with an ABI mismatch.
2. **The cuda.cmake NVTX patch is different from Step 3.** torch 2.4's
   `cuda.cmake` has a stricter check at line 69:
   ```cmake
   if(NOT TARGET CUDA::nvToolsExt)
     message(FATAL_ERROR "Failed to find nvToolsExt")
   endif()
   ```
   That fataling stops the build before our Step 3 patch (which targets
   line 173) is even reached. Replace lines 69-71 with:
   ```cmake
   if(NOT TARGET CUDA::nvToolsExt)
     add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
     target_include_directories(CUDA::nvToolsExt INTERFACE
       "C:/projects/ai/nvtx_redist/extracted/nvidia/nvtx/include")
   endif()
   ```
   This creates an INTERFACE-imported target backed by the NVTX3 headers,
   satisfying both the early check and the later `set_property(... CUDA::nvToolsExt)`
   linkage at line ~185 (which works because INTERFACE libraries propagate
   include dirs without needing an actual .lib).
3. **`spconv-cu121` upstream pin is 2.3.6 but it's been yanked from PyPI** —
   use `2.3.8` (the closest still-available release).

Stable3DGen also depends on `triton`, which has no Windows wheel. The
runner's requirements.txt deliberately omits it; if the inference path
ever requires it at runtime, we'll need a `triton-windows` fork or a
different solution. Smoke-test before assuming it works without.

#### Step 4b: model-specific local CUDA extensions (SF3D only, so far)
SF3D ships two C++/CUDA extensions inside its repo (`texture_baker/`,
`uv_unwrapper/`). They have to be installed via explicit paths after the
upstream deps, because:
- Bare `./texture_baker/` lines in a requirements file are rejected by
  pip 24+.
- Their `setup.py` imports torch at build time, which means
  `--no-build-isolation` is required.
- `wheel` + `setuptools` must already be in the venv (no isolated build env
  to bring them in).
- These env vars MUST be set in the **PowerShell** scope before invoking
  cmd, NOT inside the `cmd /c` chain — `set` inside cmd doesn't always
  propagate to pip's build subprocess:
  ```powershell
  $env:TORCH_CUDA_ARCH_LIST = "8.6"   # 3090; adjust for other GPUs
  $env:DISTUTILS_USE_SDK = "1"        # silences torch's vcvarsall warning
  ```
- pip is atomic per command: if you `pip install A B` and B fails, A is
  rolled back too. Install **one at a time** when debugging:
  ```powershell
  pip install --no-build-isolation C:\projects\ai\sf3d\stable-fast-3d\texture_baker
  pip install --no-build-isolation C:\projects\ai\sf3d\stable-fast-3d\uv_unwrapper
  ```

#### Step 5: pre-download model weights (gated — needs HF_TOKEN + license)
See each runner's `run.py` top comment for the exact HF repo and
`huggingface-cli download` command. You must accept each model's license
once on its HF page while logged in:
- <https://huggingface.co/stabilityai/TripoSR>
- <https://huggingface.co/tencent/Hunyuan3D-2>
- <https://huggingface.co/stabilityai/stable-fast-3d>
- <https://huggingface.co/Stable-X/Hi3DGen>

### 3. Run the worker
```powershell
cd C:\projects\genshape-worker-3090
.\.venv\Scripts\Activate.ps1
python worker.py
```

You should see:
```
[worker] registered as win-3090 models=[...] capacity=1
[worker] entering claim loop
```

Submit a job through the web UI (set the model to one this worker advertises)
and watch the logs.

## Adding a new model

1. `mkdir runners/<newmodel>`
2. Write `runners/<newmodel>/run.py` honouring the worker→runner contract above.
3. Write `runners/<newmodel>/requirements.txt`.
4. Repeat the per-runner setup in step 2 above.
5. Add `<newmodel>` to `WORKER_MODELS` in `.env`.
6. Add `<newmodel>` to the model dropdown on the client (separate change).
7. Restart the worker.

The server side needs **no** code change — it routes any model declared by
any registered worker, automatically.

## Operational notes

- **VRAM budget:** 24 GB on the 3090. With subprocess-per-job, only one
  runner is in VRAM at a time. Model fits easily; cold-load is ~5–30 s
  depending on model size.
- **Crash isolation:** a runner OOM / segfault dies in its own process.
  Worker keeps running, marks job failed, picks up next.
- **Cancellation:** the user clicks cancel → server flips `requestCancel`
  → next worker heartbeat (every 10 s) sees it → worker SIGTERMs the
  runner subprocess (SIGKILL after 5 s if it ignores).
- **Re-registration:** if the API server restarts, the worker's registry
  entry vanishes. Next `/claim` returns 404 → worker re-registers
  automatically and resumes.
