# Step by Step: Container and venv on Hypatia

Checklist for going from `sbi_comp_opt_hypatia.def` to a working GPU
environment on Hypatia, then using it every day. For why each step is done
this way, and for fixing errors, see `apptainer-rootless-reference.md`.

| Step | What | Where | Time |
|---|---|---|---|
| 0 | Set temp directories | Login node | 1 min |
| 1 | Check the input files | Login node | 2 min |
| 2 | Build the image (`.sif`) | Login node | ~60–90 min |
| 3 | Create the overlay (`.img`) | Login node | 1–2 min |
| 4 | Lock the dependencies | Login node, **VS Code terminal** | a few min |
| 5 | Install the venv into the overlay | **GPU compute node** | ~30–45 min |
| 6 | Check the GPU | GPU compute node | 5 min |
| 7 | Register the Jupyter kernel (optional) | GPU compute node | 1 min |
| 8 | Update SLURM scripts | — | — |

## Hypatia quirks to know first

The login node and the compute nodes behave differently, and each one causes
a failure if you run a step in the wrong place:

| | Login node (`hypatia-login`) | GPU compute nodes (`compute-gpu-0-*`) |
|---|---|---|
| CPU | Xeon E5640, **no AVX** → JAX can't import | Has AVX/AVX2 |
| User namespaces | No (`max_user_namespaces = 0`) | Yes |
| Mounting the `.img` overlay | Works as normal (setuid mode) | **Needs `--userns`** (setuid mounting of `.img` files is disabled) |
| VS Code GitHub login (for private `s2ai`) | Available in VS Code terminals | Not available |
| Internet | Yes | Yes |
| GPU | No | V100 / A100 / L40S |

So:

- **Build the image and lock** on the login node.
- **Install and run anything that imports JAX** on a compute node, always with
  `--userns`.

---

## Step 0 — Set temporary directories

**Do this in every new shell before steps 2–5.**

```bash
cd $HOME/sbi-compression
mkdir -p $HOME/.apptainer/tmp $HOME/.apptainer/cache $HOME/tmp
export APPTAINER_TMPDIR=$HOME/.apptainer/tmp     # build unpacks the image here (~10 GB)
export APPTAINER_CACHEDIR=$HOME/.apptainer/cache # downloaded image layers, reused on rebuild
export TMPDIR=$HOME/tmp                          # temp files for the s2fft/CMake build
```

**Why:** the login node's `/tmp` has only ~7 GB free. Without these, the build
and the s2fft compile fail with `No space left on device`.

**Tip:** to avoid retyping, add the three `export` lines to `~/.bashrc`.

---

## Step 1 — Check the input files

```bash
ls sbi_comp_opt_hypatia.def config_hypatia/pyproject.toml
grep -n '^From:' sbi_comp_opt_hypatia.def
grep -n 'no-install-recommends\|add-apt-repository\|software-properties' sbi_comp_opt_hypatia.def
grep -n 'torch\|pytorch-cu' config_hypatia/pyproject.toml
ls config_hypatia/uv.lock 2>/dev/null && echo "NOTE: lock present"
```

**Expected:**

- `From: nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04`
- Two lines with `--no-install-recommends`, and **no** active
  `add-apt-repository` or `software-properties-common` lines (commented-out
  ones are fine)
- `torch==...` plus a `pytorch-cu126` index and `torch = [{ index = "pytorch-cu126" }]`
- A `config_hypatia/uv.lock` is fine if it was made on Hypatia. If it was
  copied from another cluster, delete it: `rm config_hypatia/uv.lock`.
  Step 4 makes a new one.

---

## Step 2 — Build the image

On the login node:

```bash
apptainer build sbi_container_hypatia.sif sbi_comp_opt_hypatia.def 2>&1 | tee build_hypatia.log
```

**What you'll see:**

1. `INFO: The %post section will be run under fakeroot`. That's expected on Hypatia.
2. `Copying blob ...`, then `info unpack layer ...` lines for ~25 minutes.
   Many `warn xattr ... ENOTSUP` lines are harmless.
3. `Running post scriptlet`, then the apt, uv and Python installs from `%post`.
4. `INFO: Creating SIF file...`. This compresses ~7 GB into a ~4 GB image and
   can sit there for **an hour** with no output. It isn't stuck: run
   `ls -lh sbi_container_hypatia.sif` in another terminal and the size grows.
5. `INFO: Build complete: sbi_container_hypatia.sif`.

**Check it:**

```bash
apptainer sif list sbi_container_hypatia.sif | grep FS          # must say amd64, not arm64
apptainer exec sbi_container_hypatia.sif python3.13 --version   # Python 3.13.x
apptainer exec sbi_container_hypatia.sif uv --version
apptainer exec sbi_container_hypatia.sif nvcc --version         # release 12.2
```

**If it fails:**

- **`groupadd: failure while writing changes to /etc/group` / `dpkg: error
  processing package systemd`.** The build runs under a limited "fake root"
  that can't create system users or groups, and some package pulled in
  `systemd`. The `.def` avoids this by using `--no-install-recommends` and
  adding the deadsnakes PPA by hand instead of `add-apt-repository` (see
  section 3 of `apptainer-rootless-reference.md`). If a new package
  triggers it, look at `The following additional packages will be installed:`
  in the log to see what brought it in.
- **A fakeroot error before `%post` runs.** Build on a machine where you have
  `sudo`, **x86_64 only**, and copy the image over.

A failed build cleans up after itself. Rerunning reuses the cached layers, so
nothing downloads twice, but the ~25 minute unpack happens again.

---

## Step 3 — Create the overlay

On the login node:

```bash
apptainer overlay create --size 20480 --create-dir /opt/venv sandbox_overlay_hypatia.img
```

- `--size 20480` makes a 20 GB file. The installed venv is 7.7 GB. An
  overlay can't easily be resized later, so this leaves room to grow.
- `--create-dir /opt/venv` creates `/opt/venv` inside the overlay, owned by
  you. Without it you'd get `Permission denied`, because `/opt` in the image
  is owned by root.
- Use a **new** file name. The old `sandbox_overlay.img` holds an aarch64
  venv from Isambard and won't work here.

---

## Step 4 — Lock the dependencies

On the login node, **from a VS Code terminal**:

```bash
apptainer exec --overlay sandbox_overlay_hypatia.img \
  -B $HOME/sbi-compression:/opt/sbi-compression \
  -B /run/user/$(id -u) \
  sbi_container_hypatia.sif \
  uv lock --project /opt/sbi-compression/config_hypatia
```

This resolves every package to an exact version and writes
`config_hypatia/uv.lock`. It also downloads `s2ai` into uv's cache
(`~/.cache/uv`, in your home), which step 5 reuses.

**Why `-B /run/user/$(id -u)`:** `s2ai` comes from a private GitHub repo, so
git needs a GitHub login to fetch it. Hypatia has no GitHub key; the login
comes from VS Code, which answers git's credential requests through a socket
in `/run/user/<uid>/`. Apptainer doesn't mount that directory by default, so
without the bind uv fails with `Repository not found` and
`connect ENOENT /run/user/.../vscode-git-....sock`. That socket only exists on
the login node, which is why locking happens here.

**If it fails with `connect ECONNREFUSED /run/user/.../vscode-git-....sock`:**
the terminal is pointing at a dead socket from an old VS Code session (this
happens after VS Code reconnects). Either open a new VS Code terminal
(Terminal → New Terminal), or point this one at the live socket:

```bash
ss -xl | grep "/run/user/$(id -u)/vscode-git"     # the socket VS Code is listening on now
export VSCODE_GIT_IPC_HANDLE=/run/user/$(id -u)/vscode-git-<id>.sock   # use the path printed above
```

Then rerun the lock.

**Check torch came from the CUDA 12.6 index (important):**

```bash
grep -A3 '^name = "torch"' config_hypatia/uv.lock
grep -c 'cu13' config_hypatia/uv.lock      # should be 0
```

Expected: `version = "2.13.0+cu126"` and
`source = { registry = "https://download.pytorch.org/whl/cu126" }`. If it says
`pypi.org`, torch would be the CUDA 13 build, which doesn't run on Hypatia's
CUDA 12.2 driver. Check `[tool.uv.sources]` in `pyproject.toml` and re-lock.

---

## Step 5 — Install the venv into the overlay (GPU compute node)

### 5a — Get a compute node and check it

```bash
srun -p GPU --gres=gpu:a100:1 --cpus-per-task=8 --time=02:00:00 --pty bash
grep -o -w -m1 avx2 /proc/cpuinfo       # must print "avx2"
curl -sI https://pypi.org | head -1     # should print "HTTP/1.1 200 OK"
cd $HOME/sbi-compression
mkdir -p $HOME/tmp && export TMPDIR=$HOME/tmp
```

**Why a compute node:** building s2fft runs
`python -c "from jax import ffi; print(ffi.include_dir())"` to find the XLA
headers. On the login node JAX refuses to start (`This version of jaxlib was
built using AVX instructions, which your CPU ... do not support`), so the
header path comes back empty and the compile fails with
`fatal error: xla/ffi/api/api.h: No such file or directory`.

### 5b — Install

```bash
apptainer exec --userns --overlay sandbox_overlay_hypatia.img \
  -B $HOME/sbi-compression:/opt/sbi-compression \
  --env CUDAARCHS="70;80;89" \
  --env GIT_ASKPASS= \
  sbi_container_hypatia.sif \
  bash -c 'uv sync --project /opt/sbi-compression/config_hypatia --frozen && uv cache clean' 2>&1 | tee sync_hypatia.log
```

What each extra flag fixes:

- **`--userns`.** Without it, compute nodes refuse the overlay:
  `configuration disallows users from mounting extfs in setuid mode, try --userns`.
  With it, Apptainer mounts the `.img` through a user namespace instead.
- **`--env CUDAARCHS="70;80;89"`.** Without it, CMake builds s2fft's GPU code
  only for `CUDA_ARCHITECTURES: 52` (old Maxwell GPUs), because CMake's
  default wins over s2fft's own setting. 70, 80 and 89 are Hypatia's V100,
  A100 and L40S.
- **`--env GIT_ASKPASS=`.** Switches off VS Code's credential helper, which
  can't work on a compute node. s2ai comes from the uv cache filled in step 4.
  The warning `Environment variable GIT_ASKPASS already has value []` is
  expected.

Other notes:

- The overlay is mounted **writable** (no `:ro`). Make sure no other
  container or job is using it.
- `uv cache clean` only runs if the sync succeeds, and frees ~8 GB.
- If uv tries to fetch s2ai from GitHub and fails, the cache was cleared
  since step 4. Rerun step 4 on the login node, then this step.

### 5c — Check it (same session)

```bash
apptainer exec --userns --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif \
  python -c "import sys, torch, jax, s2fft, s2ai, sbi_compression; print(sys.executable); print('torch', torch.__version__); print('jax', jax.__version__)"
```

Expected: `/opt/venv/bin/python`, `torch 2.13.0+cu126`, `jax 0.11.x`.

To confirm s2fft was built for the right GPUs:

```bash
apptainer exec --userns --overlay sandbox_overlay_hypatia.img:ro sbi_container_hypatia.sif \
  bash -c 'cuobjdump --list-elf /opt/venv/lib/python3.13/site-packages/s2fft_lib/_s2fft.abi3.so | grep -o "sm_[0-9]*" | sort -u'
```

The list must include `sm_70`, `sm_80` and `sm_89`.

---

## Step 6 — Check the GPU

In the same compute-node session (or a new `srun` as in 5a):

```bash
nvidia-smi       # top right: "CUDA Version: 12.2"
apptainer exec --userns --nv --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif \
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
apptainer exec --userns --nv --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif \
  python -c "import jax, jax.numpy as jnp; print(jax.devices()); print((jnp.ones((1000,1000)) @ jnp.ones((1000,1000))).sum())"
```

Expected:

- torch: `2.13.0+cu126 12.6 True <GPU name>`
- JAX: `[CudaDevice(id=0)]`, then `1e+09`. A warning that the driver is older
  than `ptxas` and parallel compilation is disabled is normal.

Leave the session with `exit`.

---

## Step 7 — Register the Jupyter kernel (optional)

On a compute node:

```bash
apptainer run --userns --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif \
  -c "print('kernel registered')"
```

`%runscript` registers a kernel called "Python 3.13 (SBI Container)" in
`~/.local/share/jupyter/kernels/`. For the kernel to actually launch inside
the container, its `kernel.json` has to call `apptainer exec --userns ...`
rather than `/opt/venv/bin/python` directly, and Jupyter itself has to run
on a compute node (JAX can't import on the login node).

---

## Step 8 — Update SLURM scripts for Hypatia

Ready-made versions exist for every job script: `examples/*/<name>_hypatia.sh`
(Hypatia) and `examples/*/<name>_isambard.sh` (Isambard). Submit from the repo
root, e.g. `sbatch examples/sbi/NLE_s2MSE_hypatia.sh`. To convert a new
Isambard script, change the `#SBATCH` lines and the `apptainer exec` part:

```bash
# Isambard (old)
apptainer exec --nv --overlay sandbox_overlay.img:ro \
    -B /home/u6pf/brianycc.u6pf/sbi-compression:/opt/sbi-compression \
    -B /scratch:/scratch \
    -B /projects:/projects \
    sbi_container.sif /opt/venv/bin/python \
    examples/NLE_s2.py ...

# Hypatia (new)
apptainer exec --userns --nv --overlay sandbox_overlay_hypatia.img:ro \
    -B $HOME/sbi-compression:/opt/sbi-compression \
    -B /share/lustre/brianycc \
    sbi_container_hypatia.sif /opt/venv/bin/python \
    examples/NLE_s2.py ...
```

- Add **`--userns`**, and use the new image and overlay names. Keep `:ro`.
- `/scratch` and `/projects` don't exist on Hypatia. Remove those binds,
  bind wherever your data lives instead (e.g. `/share/lustre/brianycc`), and
  change the data paths in the script arguments.
- Use `#SBATCH --partition=GPU` and request a GPU type. Hypatia's GPU nodes:

  | Nodes | GPUs | CPUs |
  |---|---|---|
  | `compute-gpu-0-[0-1]` | 10 × V100 (`gpu:v100:N`) | 48 |
  | `compute-gpu-0-[2-4]` | 4–5 × A100 (`gpu:a100:N`) | 48–64 |
  | `compute-gpu-0-5` | 10 × L40S | 128 |

  `--cpus-per-task=72`, `--mem=256G` and `--gres=gpu:4` were Isambard GH200
  values. The GPU partition has a 2-day time limit.
- Relative paths like `examples/NLE_s2.py` still work if you `sbatch` from
  `$HOME/sbi-compression`, because the container starts in the host's
  current directory.

Minimal template:

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

cd $HOME/sbi-compression
hostname
nvidia-smi --list-gpus

apptainer exec --userns --nv --overlay sandbox_overlay_hypatia.img:ro \
    -B $HOME/sbi-compression:/opt/sbi-compression \
    sbi_container_hypatia.sif /opt/venv/bin/python \
    examples/my_script.py
```

---

## Day-to-day use

Always on a **compute node** (`srun` or `sbatch`), never the login node.

**Interactive shell:**

```bash
srun -p GPU --gres=gpu:1 --time=04:00:00 --pty bash
cd $HOME/sbi-compression
apptainer shell --userns --nv --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif
```

**Run one script:**

```bash
apptainer exec --userns --nv --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif \
  python /opt/sbi-compression/my_script.py
```

**Batch job:** `sbatch examples/my_job.sh`, using the template in step 8.

**Rules:**

- `--userns` on compute nodes, every time.
- `:ro` on the overlay for anything except installing packages. Many jobs can
  share a read-only overlay; only one container can have it open writable.
- Always bind the repo at `/opt/sbi-compression`, because `sbi_compression`
  is installed editable from there.
- Always pass `--nv` when you want the GPU.
- Don't use `--containall`/`-c`. `/tmp` would shrink to 64 MB.
- Edit code on the host as normal; changes show up in the container at once.

---

## Changing Python packages

1. Edit `config_hypatia/pyproject.toml`.
2. Make sure no jobs or shells are using the overlay.
3. **Login node, VS Code terminal:** set the step 0 variables and re-lock
   (step 4). Check torch still comes from `cu126`.
4. **Compute node:** re-sync (step 5b), with the overlay mounted **writable**.
5. Re-run the step 5c and step 6 checks.

`uv sync` also removes packages that are no longer in the lock.

## When to rebuild the image instead

Only when something **outside** the venv changes: apt packages, the CUDA base
image, or the Python version. Then:

1. Rebuild the SIF (step 2), using a new name if jobs are still using the old one.
2. If the Python version changed, recreate the overlay (step 3) and reinstall
   (steps 4–5). Otherwise the existing overlay keeps working.

---

## Troubleshooting summary

| Error | Where | Fix |
|---|---|---|
| `groupadd: failure while writing changes to /etc/group` (systemd) | Build | `--no-install-recommends`; no `software-properties-common` (step 2) |
| `Creating SIF file...` for an hour | Build | Normal; check that the `.sif` size is growing |
| `Repository not found` + `connect ENOENT .../vscode-git-....sock` | Lock | Add `-B /run/user/$(id -u)`; use a VS Code terminal (step 4) |
| `Repository not found` + `connect ECONNREFUSED .../vscode-git-....sock` | Lock | New VS Code terminal, or export `VSCODE_GIT_IPC_HANDLE` (step 4) |
| `jaxlib was built using AVX instructions` / `xla/ffi/api/api.h: No such file` | Install or run on login node | Use a compute node (step 5a) |
| `configuration disallows users from mounting extfs in setuid mode, try --userns` | Compute node | Add `--userns` |
| `CUDA_ARCHITECTURES: 52` in the s2fft build | Install | Add `--env CUDAARCHS="70;80;89"` (step 5b) |
| `ModuleNotFoundError: No module named 'torch'` | Anywhere | `uv sync` didn't finish; check `sync_hypatia.log` and rerun step 5 |
| `torch.cuda.is_available()` → `False` | Compute node | Check torch is `+cu126` (step 4); pass `--nv` |

---

## Cleaning up after it works

```bash
rm -rf $HOME/tmp/* $HOME/.apptainer/tmp/*   # build and install leftovers
apptainer cache clean                       # cached image layers, if you won't rebuild soon
```

**Keep the Isambard files** (`sbi_container.sif`, `sandbox_overlay.img`,
`sbi_comp_isambard.def`, `config_isambard/`) if you'll use Isambard again.
They're arm64 and won't run on Hypatia, but they're the working Isambard
environment. Copy them back to Isambard rather than rebuilding there. They
take ~24 GB of Hypatia home space; if space gets tight, move them to
`/share/lustre/brianycc/` or keep them only on Isambard.
