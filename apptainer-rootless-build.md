# Building Apptainer Containers Without Root Access

How to build and use the `sbi-compression` container on a cluster where you
have no root, no `sudo`, and no unprivileged user namespaces (Hypatia is like
this). It covers why the usual approach fails, the rules the `.def` file has
to follow, and the full sequence from building the image to daily use.

Files involved:

- `sbi_comp_opt_hypatia.def` — container definition (OS, CUDA toolkit, Python 3.13, uv)
- `config_hypatia/pyproject.toml` — Python dependencies for the venv
- `sbi_container_hypatia.sif` — built image (read-only)
- `sandbox_overlay_hypatia.img` — writable overlay holding the venv at `/opt/venv`

## Check what the cluster allows

```bash
apptainer --version
cat /proc/sys/user/max_user_namespaces   # 0 = unprivileged user namespaces disabled
grep $USER /etc/subuid                   # empty = no subuid mapping
uname -m                                 # x86_64 or aarch64: the image is built for this
nvidia-smi                               # on a GPU node: "CUDA Version" is the max CUDA the driver supports
```

If `max_user_namespaces` is `0` and you have no `/etc/subuid` entry,
`apptainer build` falls back to **fakeroot-only mode**. You will see:

```
INFO:    User not listed in /etc/subuid, trying root-mapped namespace
INFO:    Could not start root-mapped namespace
INFO:    The %post section will be run under fakeroot
```

In this mode `%post` only *pretends* to be root (via `LD_PRELOAD`). Most
`apt-get install` steps work, but anything that writes system account files
does not.

## Rules for a `.def` that builds without root

1. **Don't install packages that create system users or groups.** Their
   install scripts call `groupadd`/`useradd`, which fail under fakeroot:
   ```
   groupadd: failure while writing changes to /etc/group
   dpkg: error processing package systemd (--configure)
   ```
   The usual cause is `systemd` and `dbus`, which get pulled in indirectly.

2. **Always use `apt-get install --no-install-recommends`.** Without it, apt
   also installs "recommended" extras (packagekit, unattended-upgrades,
   openssh-client, …) that drag in systemd. List anything you actually need
   explicitly (e.g. `ca-certificates` for HTTPS downloads).

3. **Don't use `add-apt-repository` / `software-properties-common`.** It
   depends on packagekit → policykit → systemd. Add a PPA by hand instead.
   A PPA (Personal Package Archive) is an extra Ubuntu package repository
   hosted on Launchpad; adding one is just a signing key plus a sources line:
   ```bash
   mkdir -p /etc/apt/keyrings
   curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
       -o /etc/apt/keyrings/deadsnakes.asc
   echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.asc] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) main" \
       > /etc/apt/sources.list.d/deadsnakes.list
   apt-get update && apt-get install -y --no-install-recommends python3.13 python3.13-dev python3.13-venv
   ```
   The key fingerprint is the deadsnakes PPA's signing key as published on
   Launchpad. (Alternative with no PPA at all: `uv python install 3.13`.)

4. **Don't build the venv into the image.** Anything `%post` creates under
   `/opt` is root-owned, so you can't `pip`/`uv` install into it later as a
   normal user. Keep the image to OS + CUDA + Python + uv, and put the venv in
   an overlay (below).

5. **The image architecture follows the build host.** `Bootstrap: docker`
   pulls the layer matching the machine you build on. An image built on an
   aarch64 cluster (e.g. Isambard) will not run on an x86_64 one (Hypatia).
   Check with `apptainer sif list <image>.sif | grep FS`.

6. **Match the CUDA base image to the cluster's driver.** Use a base image no
   newer than the driver's CUDA version (Hypatia: driver supports CUDA 12.2 →
   `nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04`). pip-installed CUDA 12.x
   wheels (e.g. `torch+cu126`, `jax[cuda12]`) still run on a 12.2 driver, but
   CUDA **13** wheels do not. PyTorch's default PyPI wheel is now CUDA 13, so
   pin torch to a `cu12x` index in `pyproject.toml` and re-lock per cluster.

7. **Don't override the pip CUDA libraries.** Leave `LD_LIBRARY_PATH` and
   `XLA_FLAGS=--xla_gpu_cuda_data_dir` unset in `%environment`, so JAX/torch
   use the CUDA libraries from their own `nvidia-*` wheels rather than the
   image's toolkit.

## Harmless messages during a fakeroot build

- `warn xattr{...} ignoring ENOTSUP on setxattr "user.rootlesscontainers"` — NFS doesn't support xattrs.
- `WARNING: The --fix-perms option modifies the filesystem permissions` — expected in this mode.
- `ln: failed to create symbolic link '/etc/resolv.conf': Device or resource busy` — Apptainer bind-mounts that file.
- `debconf: delaying package configuration, since apt-utils is not installed`.

---

## Full procedure (Hypatia)

| Steps | Where | Why |
|---|---|---|
| 0–3 | Login node | Needs internet (Docker image, apt, PyPI, PyTorch index, GitHub). No GPU needed. |
| 4 and daily use | GPU compute node | The login node has no GPU. |

The install compiles s2fft and is CPU-heavy. If login-node use is restricted
and compute nodes have internet (`srun ... curl -sI https://pypi.org`), run
steps 2–3 in an interactive job instead.

### Step 0 — Point temporary files at home (every new shell)

The login node's `/tmp` is small (~7 GB free on Hypatia); the build unpacks
~10 GB and the s2fft compile also writes temp files.

```bash
cd $HOME/sbi-compression
mkdir -p $HOME/.apptainer/tmp $HOME/.apptainer/cache $HOME/tmp
export APPTAINER_TMPDIR=$HOME/.apptainer/tmp     # build unpacks the image here
export APPTAINER_CACHEDIR=$HOME/.apptainer/cache # cached Docker layers (reused on rebuild)
export TMPDIR=$HOME/tmp                          # temp files for uv/CMake builds, passed into the container
```

Unpacking onto NFS home is slow (~25 min before `%post` starts); that is normal.

### Step 1 — Build the image (login node)

```bash
apptainer build sbi_container_hypatia.sif sbi_comp_opt_hypatia.def
```

Check it:

```bash
apptainer sif list sbi_container_hypatia.sif | grep FS      # must say amd64 on Hypatia
apptainer exec sbi_container_hypatia.sif python3.13 --version
apptainer exec sbi_container_hypatia.sif uv --version
```

If the build still fails and you can't fix it, build on a machine where you
have `sudo` (same architecture) and copy the image over:

```bash
sudo apptainer build sbi_container_hypatia.sif sbi_comp_opt_hypatia.def
rsync -avP sbi_container_hypatia.sif hypatia:~/sbi-compression/
```

### Step 2 — Create the overlay (login node)

```bash
apptainer overlay create --size 20480 --create-dir /opt/venv sandbox_overlay_hypatia.img
```

20 GB ext3 image. `--create-dir /opt/venv` makes that directory owned by you
inside the overlay; without it you'd get "permission denied" writing under the
root-owned `/opt`.

### Step 3 — Lock and install the venv into the overlay (login node)

Generate a lock file for this cluster (don't reuse another cluster's lock):

```bash
apptainer exec --overlay sandbox_overlay_hypatia.img \
  -B $HOME/sbi-compression:/opt/sbi-compression \
  sbi_container_hypatia.sif \
  uv lock --project /opt/sbi-compression/config_hypatia
```

Check torch resolved to the CUDA 12 build before installing:

```bash
grep -A3 '^name = "torch"' config_hypatia/uv.lock
# want:  source = { registry = "https://download.pytorch.org/whl/cu126" }
# not:   source = { registry = "https://pypi.org/simple" }
```

Install:

```bash
apptainer exec --overlay sandbox_overlay_hypatia.img \
  -B $HOME/sbi-compression:/opt/sbi-compression \
  sbi_container_hypatia.sif \
  bash -c 'uv sync --project /opt/sbi-compression/config_hypatia --frozen && uv cache clean'
```

`uv sync` creates `/opt/venv` itself (from `UV_PROJECT_ENVIRONMENT` and
`UV_PYTHON` in `%environment`). `uv cache clean` frees the duplicate copy of
the wheels in `~/.cache/uv`.

Check (CPU only):

```bash
apptainer exec --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif \
  python -c "import sys, torch, jax, s2fft, sbi_compression; print(sys.executable, torch.__version__, jax.__version__)"
```

Expect `/opt/venv/bin/python` and a torch version ending in `+cu126`.

Optionally register the Jupyter kernel (via `%runscript`):

```bash
apptainer run --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif -c "print('kernel registered')"
```

### Step 4 — GPU check (compute node)

```bash
srun --gres=gpu:1 --time=00:20:00 --pty bash      # add -p <gpu-partition> as required
cd $HOME/sbi-compression
nvidia-smi
apptainer exec --nv --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif \
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
apptainer exec --nv --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif \
  python -c "import jax, jax.numpy as jnp; print(jax.devices()); print((jnp.ones((1000,1000)) @ jnp.ones((1000,1000))).sum())"
```

Expected: torch prints `2.13.0+cu126 12.6 True <GPU name>`; jax lists
`CudaDevice(id=0)`. A JAX warning that the driver is older than ptxas and
parallel compilation is disabled is expected and harmless.

---

## Day-to-day use

Interactive shell:

```bash
apptainer shell --nv --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif
```

Run a script (e.g. inside a SLURM job):

```bash
apptainer exec --nv --overlay sandbox_overlay_hypatia.img:ro \
  -B $HOME/sbi-compression:/opt/sbi-compression sbi_container_hypatia.sif \
  python /opt/sbi-compression/my_script.py
```

Notes:

- **Always mount the overlay `:ro` for jobs.** A writable ext3 overlay can only
  be open in one container at a time; read-only lets many jobs share it.
- **Always bind the repo at `/opt/sbi-compression`.** `sbi_compression` is
  installed editable from that path.
- **Temp files:** host `/tmp` is mounted into the container automatically
  (node-local on compute nodes). Don't use `--containall`/`-c`: `/tmp` then
  becomes a 64 MB in-memory directory and GPU kernel compiles run out of space.
  Compiled JAX kernels persist in `~/.cache/jax` (`JAX_COMPILATION_CACHE_DIR`).

## Updating packages

1. Edit `config_hypatia/pyproject.toml`.
2. Make sure no jobs are using the overlay.
3. From the login node, with the Step 0 variables set, rerun Step 3 (lock, check
   torch, sync), mounting the overlay **without** `:ro`.

The image only needs rebuilding if system packages, the CUDA base image or the
Python version change.
