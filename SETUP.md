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
For each model under `runners/`:
```powershell
cd C:\projects\genshape-worker-3090\runners\<model>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Then pre-download weights to C:\projects\ai\<model>\ — see each runner's
# run.py top comment for the exact HF repo + huggingface-cli command.
deactivate
```

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
