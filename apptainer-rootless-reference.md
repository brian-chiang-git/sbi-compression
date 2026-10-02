# Apptainer Without Root: Detailed Reference

In-depth companion to `apptainer-rootless-build.md` (the short summary) and
`apptainer-hypatia-step-by-step.md` (the exact commands in order). This file
explains how the container setup fits together, why building without root is
limited, the rules the `.def` file has to follow, what each setting does, and
how to diagnose common errors.

Files involved:

| File | What it is |
|---|---|
| `sbi_comp_opt_hypatia.def` | Container definition: Ubuntu 22.04, CUDA 12.2 toolkit, Python 3.13, uv |
| `config_hypatia/pyproject.toml` | Python dependencies for the venv, including the torch CUDA index |
| `config_hypatia/uv.lock` | Exact resolved versions, generated **on Hypatia** |
| `sbi_container_hypatia.sif` | Built image (read-only) |
| `sandbox_overlay_hypatia.img` | Writable ext3 overlay holding the venv at `/opt/venv` |

---

## 1. How the pieces fit together

A running container is three layers stacked on top of each other:

```
 inside the container                         comes from
 ─────────────────────────────────────────    ─────────────────────────────────────
 /opt/sbi-compression/  (your code)       ←   -B $HOME/sbi-compression:/opt/sbi-compression
 /opt/venv/             (Python packages) ←   --overlay sandbox_overlay_hypatia.img
 /usr, /etc, /usr/local/cuda, python3.13  ←   sbi_container_hypatia.sif
 /tmp, $HOME                              ←   host (mounted automatically)
```

- **The SIF** is a compressed, read-only squashfs image. It is built once from
  the `.def` and holds everything that needs root to install: apt packages,
  the CUDA toolkit, Python 3.13 and uv.
- **The overlay** is an ext3 filesystem in a single file. It is layered on
  top of the SIF, so anything written to `/opt/venv` lands in the overlay
  rather than the (read-only) SIF. It holds the Python venv.
- **The bind mount** maps your repo on the host into the container. Code edits
  on the host are visible immediately; nothing about the code is baked in.

Why split the venv out of the image:

1. **Permissions.** Anything `%post` creates is owned by root. As a normal
   user you could never `uv`/`pip` install into it afterwards.
2. **Rebuild time.** Building the SIF takes ~30+ minutes on NFS. Changing a
   Python package only means re-running `uv sync` into the overlay.
3. **Per-cluster environments.** The same `.def` style works everywhere, and
   each cluster gets its own lock file and overlay matched to its GPU driver.

---

## 2. Why building without root is limited

Running `%post` needs root inside the container (to write `/usr`, run
`apt-get`, etc.). Apptainer gets that in one of these ways, trying each in turn:

| Mode | Needs | Capability |
|---|---|---|
| Real root | `sudo apptainer build` | Everything |
| Full fakeroot | Unprivileged user namespaces **and** an `/etc/subuid` entry | Nearly everything |
| Root-mapped namespace | Unprivileged user namespaces | Most things |
| **fakeroot-only** | setuid Apptainer + `/usr/bin/fakeroot` | Limited: fakes root via `LD_PRELOAD` |

Check which applies:

```bash
apptainer --version
cat /proc/sys/user/max_user_namespaces   # 0 = user namespaces disabled
grep $USER /etc/subuid                   # empty = no subuid mapping
which fakeroot                           # needed for fakeroot-only mode
uname -m                                 # the image is built for this architecture
```

On Hypatia (`max_user_namespaces = 0`, no subuid entry), `apptainer build`
falls back to fakeroot-only mode and prints:

```
INFO:    User not listed in /etc/subuid, trying root-mapped namespace
INFO:    Could not start root-mapped namespace
INFO:    The %post section will be run under fakeroot
INFO:    Using --fix-perms because building from a definition file
INFO:     without either root user or unprivileged user namespaces
```

In this mode, programs in `%post` are told they are root, but the kernel still
treats them as you. File installs work; anything that edits system account
databases (`/etc/group`, `/etc/passwd`, `/etc/shadow`) fails. That rules out
packages whose install scripts create system users or groups, most notably
`systemd` and `dbus`.

**Pulling** an image (`apptainer build x.sif docker://...`) and **running**
one never need root; only building from a `.def` with `%post` does.

---

## 3. Rules for a `.def` that builds without root

### 3.1 Never install packages that create users or groups

The failure looks like this:

```
Setting up systemd (249.11-0ubuntu3.22) ...
groupadd: failure while writing changes to /etc/group
addgroup: `/sbin/groupadd -g 101 systemd-journal' returned error code 10. Exiting.
dpkg: error processing package systemd (--configure):
E: Sub-process /usr/bin/dpkg returned an error code (1)
FATAL:   While performing build: while running engine: exit status 100
```

Usual culprits: `systemd`, `dbus`, `polkitd`, `openssh-server`, database
servers. These are almost never wanted in an HPC container; they get pulled
in indirectly.

To find what pulls a package in, look at the apt output above the failure
(`The following additional packages will be installed: ...`), or inside a
built image run `apt-cache rdepends --installed <package>`.

### 3.2 Always use `--no-install-recommends`

By default apt also installs "recommended" packages. For this image that
brought in packagekit, unattended-upgrades, openssh-client, policykit and
through them systemd. With `--no-install-recommends` apt installs only hard
dependencies, so list anything you actually need explicitly:

```bash
# ca-certificates: HTTPS for curl, git and uv (previously only arrived as a "recommend")
apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget curl git cmake ninja-build g++ pkg-config \
    libproj-dev proj-bin libgeos-dev libfreetype6-dev libpng-dev \
    libopenblas-dev gfortran
```

### 3.3 Add PPAs by hand, not with `add-apt-repository`

A **PPA (Personal Package Archive)** is an extra Ubuntu package repository
hosted on Launchpad. Ubuntu 22.04 only ships Python 3.10; the **deadsnakes**
PPA provides `python3.13`, `python3.13-dev` (headers, needed to compile
s2fft) and `python3.13-venv`.

`add-apt-repository` needs `software-properties-common`, which depends on
packagekit → policykit → systemd. Adding a PPA is really just two files, so
write them directly:

```bash
# 1. The PPA's signing key, so apt can verify the packages
mkdir -p /etc/apt/keyrings
curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
    -o /etc/apt/keyrings/deadsnakes.asc

# 2. Where the repository is ($VERSION_CODENAME = "jammy" on Ubuntu 22.04)
echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.asc] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) main" \
    > /etc/apt/sources.list.d/deadsnakes.list

apt-get update && apt-get install -y --no-install-recommends \
    python3.13 python3.13-dev python3.13-venv
```

- `F23C5A6CF475977595C89F51BA6932366A755776` is the deadsnakes signing key
  fingerprint as published on Launchpad
  (`https://api.launchpad.net/1.0/~deadsnakes/+archive/ubuntu/ppa`).
- apt accepts an ASCII-armored `.asc` key directly, so `gpg` isn't needed.
- For another PPA, get its fingerprint from its Launchpad page and replace
  the key URL, file name and `deb` line.

apt installs to fixed paths chosen by the packager (Debian layout,
`--prefix=/usr`): the interpreter is `/usr/bin/python3.13`, the standard
library `/usr/lib/python3.13/`, headers `/usr/include/python3.13/`. Ubuntu's
own `python3` (3.10) stays at `/usr/bin/python3`, which is why uv is pointed
at 3.13 explicitly (`UV_PYTHON`).

Alternative with no PPA: uv can download a standalone Python
(`uv python install 3.13`). That would need `UV_PYTHON_INSTALL_DIR` set to a
path in the image and `UV_PYTHON` changed to match.

### 3.4 Don't create the venv in `%post`

Keep `%post` to system packages, uv and Python. If the SIF contains a
root-owned `/opt/venv`, the overlay can't make it writable for you. The
overlay is created with `--create-dir /opt/venv`, which makes that directory
owned by you, and `uv sync` fills it.

### 3.5 The image architecture follows the build host

`Bootstrap: docker` with a multi-arch image (like `nvcr.io/nvidia/cuda`)
pulls the layer matching the machine you build on. `uname -m` in `%post`
only reports it. An image built on an aarch64 cluster (Isambard, GH200) will
not run on an x86_64 cluster (Hypatia), and neither will an overlay whose
venv was installed there.

```bash
apptainer sif list <image>.sif | grep FS    # e.g. "FS (Squashfs/*System/amd64)"
```

`apptainer build --arch <arch>` can cross-build, but only if the build host
has QEMU user emulation set up (binfmt_misc), which HPC login nodes rarely do.

### 3.6 Match CUDA to the cluster's GPU driver

`nvidia-smi` on a GPU node reports a driver version and a **"CUDA Version"**:
that is the newest CUDA the driver supports (Hypatia: 12.2, driver ~535).

- **CUDA 12.x software runs on any CUDA 12.x driver** ("minor version
  compatibility"), with one exception: the driver can't JIT-compile PTX
  produced by a newer toolkit.
- **CUDA 13 software does not run on a CUDA 12 driver** at all.

What that means for each component:

| Component | Source of CUDA | On a 12.2 driver |
|---|---|---|
| Base image toolkit (`nvcc`) | `cuda:12.2.2-devel` | Matches the driver exactly, so anything compiled with it (s2fft extensions, including PTX) runs |
| torch | `nvidia-*` wheels from the `cu126` index | Works (CUDA 12.6 < 13) |
| torch from plain PyPI | CUDA 13 wheels (`nvidia-*-cu13`, `cuda-toolkit`) | ❌ `torch.cuda.is_available()` returns `False` |
| jax `[cuda12]` | CUDA 12.9 `nvidia-*-cu12` wheels | Works; XLA warns that the driver is older than `ptxas` and disables parallel compilation (slower compiles, correct results) |

So:

- Use a base image no newer than the driver's CUDA version.
- Pin torch to a CUDA 12 build via an explicit uv index (see 4.2). There is
  no `cu122` index; `cu126` is the right choice for a 12.2 driver.
- Re-lock on each cluster. A lock made elsewhere may resolve torch from PyPI.

### 3.7 Don't point JAX/torch at the image's CUDA libraries

Leave these **commented out** in `%environment`:

```bash
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
```

JAX and torch load the CUDA libraries from their own `nvidia-*` wheels. Putting
the image's toolkit first on `LD_LIBRARY_PATH` can make them load older
libraries (e.g. cuDNN/cuBLAS version-mismatch errors), and `XLA_FLAGS` would
make XLA use the older `ptxas`. Keep `PATH` (for `nvcc`) and `CUDA_HOME`,
which s2fft's CMake build uses.

---

## 4. What the configuration does

### 4.1 `%environment` variables

These apply every time the container starts (not during `%post`).

| Variable | Value | Why |
|---|---|---|
| `PATH` | `/opt/venv/bin:/usr/local/cuda/bin:$PATH` | venv's `python` first; `nvcc` available |
| `CUDA_HOME` | `/usr/local/cuda` | CMake/s2fft find the toolkit |
| `UV_PROJECT_ENVIRONMENT` | `/opt/venv` | `uv sync` installs into the overlay, not `config_hypatia/.venv` |
| `UV_PYTHON` | `/usr/bin/python3.13` | Use the deadsnakes Python, not Ubuntu's 3.10 |
| `UV_PYTHON_DOWNLOADS` | `never` | Stop uv downloading its own Python |
| `UV_LINK_MODE` | `copy` | uv cache (`~/.cache/uv`, NFS) and venv (overlay) are different filesystems, so hardlinks can't work |
| `TMPDIR` | `${TMPDIR:-/tmp}` | Temp files for nvcc/XLA/torch; a job or the host shell can override it |
| `JAX_COMPILATION_CACHE_DIR` | `${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax}` | Compiled XLA kernels persist between jobs |

Host environment variables are passed into the container, so exporting
`TMPDIR` on the host before `apptainer exec` changes it inside too.

### 4.2 `config_hypatia/pyproject.toml` (the uv parts)

```toml
[tool.uv]
no-binary-package = ["s2fft"]          # build s2fft from source → needs g++, cmake, python3.13-dev
override-dependencies = [
    "numpy>=2.3.0",                    # replaces EVERY numpy requirement, including your own pin
    "healpy>=1.17.0",
]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true                        # only packages mapped below use this index

[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]  # torch comes from the CUDA 12.6 index, not PyPI
```

- `override-dependencies` wins over everything, so a separate `"numpy==X"`
  in `dependencies` has no effect. To pin numpy, put the pin in the override.
- `uv lock` / `uv sync` only read `pyproject.toml` and `uv.lock`. Files like
  `overrides.txt` and `resolved-requirements.txt` are only used by
  `uv pip ... --overrides/-r` and can be deleted, or regenerated with
  `uv export --project config_hypatia --format requirements-txt --no-emit-project -o config_hypatia/resolved-requirements.txt`.
- `package-dir = {"" = "../src"}` installs `sbi_compression` editable from
  `/opt/sbi-compression/src`, so the repo must always be bound at
  `/opt/sbi-compression`.

### 4.3 `uv lock` vs `uv sync`

- `uv lock` resolves every dependency to exact versions and writes `uv.lock`.
  It needs internet and doesn't install anything.
- `uv sync --frozen` installs exactly what's in `uv.lock` into
  `UV_PROJECT_ENVIRONMENT`, creating the venv if needed, and removes anything
  not in the lock. `--frozen` means "don't re-resolve"; it fails if there is
  no lock file.
- `uv cache clean` deletes downloaded wheels from `~/.cache/uv`. With
  `UV_LINK_MODE=copy` they'd otherwise be stored twice (cache + overlay),
  roughly 5–10 GB for this environment.

### 4.4 Private GitHub dependencies (s2ai)

**Issue:** `s2ai` is installed from a private GitHub repo, so git needs a
GitHub login to fetch it during `uv lock` and `uv sync`. Hypatia has no GitHub
SSH key or credential helper. The login comes from VS Code: in its terminals
it sets `GIT_ASKPASS` and `VSCODE_GIT_IPC_HANDLE`, so git asks VS Code
(signed in to your GitHub account) for credentials through a socket in
`/run/user/<uid>/`. Apptainer passes those variables into the container but
doesn't mount `/run/user`, so git can't reach VS Code and GitHub replies
`Repository not found`.

**Solution:** run `uv lock` / `uv sync` from a VS Code terminal and bind the
socket directory into the container:

```bash
apptainer exec ... -B /run/user/$(id -u) sbi_container_hypatia.sif uv lock ...
```

**If the socket is dead** (`connect ECONNREFUSED`): the terminal was opened
in an earlier VS Code session (e.g. before a reconnect) and still points at
that session's socket. Open a new VS Code terminal, or point this one at the
socket VS Code is listening on now:

```bash
ss -xl | grep "/run/user/$(id -u)/vscode-git"
export VSCODE_GIT_IPC_HANDLE=/run/user/$(id -u)/vscode-git-<id>.sock   # path from the ss output
```

This only matters for installing packages. Running code never contacts
GitHub, so SLURM jobs don't need the bind.

The socket only exists on the login node, so `uv lock` runs there. It leaves
the s2ai checkout in `~/.cache/uv` (in your home, visible from every node),
and `uv sync` on a compute node then reuses it. On the compute node, pass
`--env GIT_ASKPASS=` so git doesn't try the unreachable VS Code helper. If
`uv cache clean` has removed the checkout since the last lock, re-lock on the
login node before syncing.

### 4.5 Building s2fft: JAX must import, and CUDA architectures must be set

s2fft is built from source (`no-binary-package`). Two parts of its CMake build
can fail quietly and only show up later:

**JAX has to import during the build.** CMake runs
`python -c "from jax import ffi; print(ffi.include_dir())"` to find the XLA
headers. jaxlib needs a CPU with AVX instructions. Hypatia's login node
(Xeon E5640) has none, so JAX fails with
`This version of jaxlib was built using AVX instructions, which your CPU ...
do not support`. CMake then prints an empty `XLA include directory:` line and
the compile fails with `fatal error: xla/ffi/api/api.h: No such file or
directory`. **Solution:** run `uv sync` on a compute node (AVX2 available).
The same limit means nothing that imports JAX can run on the login node.

**The GPU architectures have to be set explicitly.** s2fft's `CMakeLists.txt`
asks for `70;80;89` via `set(CMAKE_CUDA_ARCHITECTURES ... CACHE ...)`, but
CMake has already filled that cache variable with its own default (`52`) by
then, and a `CACHE` set never overrides an existing value. The log shows
`CUDA_ARCHITECTURES: 52`, which is code for 2014-era Maxwell GPUs.
**Solution:** pass `--env CUDAARCHS="70;80;89"` to `apptainer exec`. CMake
reads `CUDAARCHS` before choosing its default. 70/80/89 are Hypatia's
V100/A100/L40S. Check the result with
`cuobjdump --list-elf .../s2fft_lib/_s2fft.abi3.so | grep -o "sm_[0-9]*" | sort -u`.

---

## 5. Storage and temporary files on Hypatia

| Location | Type | Free | Use for |
|---|---|---|---|
| `$HOME` | NFS, shared | ~236 GB (98% full overall) | SIF, overlay, uv cache, `APPTAINER_TMPDIR` |
| `/tmp` on login node | local disk | ~7 GB | Too small for the build or the s2fft compile |
| `/tmp` on compute nodes | usually node-local | varies | Runtime temp files |
| `/scratch` | — | doesn't exist on Hypatia | — |

- `apptainer build` unpacks the whole image (~10 GB) into
  `APPTAINER_TMPDIR` (default `/tmp`) and caches layers in
  `APPTAINER_CACHEDIR` (default `~/.apptainer/cache`). Point both at home.
- Unpacking onto NFS is slow: ~25 minutes before `%post` starts.
- Apptainer mounts the host `/tmp` into the container automatically
  (`mount tmp = yes`). With `--containall`/`-c`, `/tmp` becomes an in-memory
  directory capped at 64 MB (`sessiondir max size = 64`), which GPU kernel
  compiles will overflow, so don't use `-c`.
- Sizes: SIF 3.8 GB, overlay 20 GB file (7.7 GB used by the venv), uv cache
  ~8 GB until cleaned.
- The final `Creating SIF file...` step of a build compresses ~7 GB from NFS
  back onto NFS, and took about an hour with no output. Check it's progressing
  with `ls -lh <image>.sif`, which grows as it runs.

### 5.1 Login node vs compute nodes

The nodes have different hardware and different Apptainer settings:

| | Login node | GPU compute nodes |
|---|---|---|
| CPU | Xeon E5640, no AVX | AVX/AVX2 |
| `max_user_namespaces` | 0 | 15000 |
| Mounting `.img` overlays in setuid mode | Allowed | Disabled (`allow setuid-mount extfs = no`) |
| VS Code git socket | Yes | No |

Consequences:

- **On compute nodes the overlay needs `--userns`.** Without it Apptainer
  refuses: `configuration disallows users from mounting extfs in setuid mode,
  try --userns`. With `--userns`, Apptainer mounts the ext3 image with FUSE
  tools (`fuse2fs`, `fuse-overlayfs`) inside a user namespace, without
  setuid. This works read-only and writable.
- **On the login node, don't add `--userns`.** User namespaces are disabled
  there, so the normal setuid mount is the only option (and it's allowed).
- **Where each step runs:** build and `uv lock` on the login node;
  `uv sync`, testing and all real work on compute nodes.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `groupadd: failure while writing changes to /etc/group`, `dpkg: error processing package systemd` | fakeroot-only build can't create system groups | 3.1–3.3: drop `software-properties-common`, add `--no-install-recommends` |
| `No space left on device` during build or `uv sync` | Writing to the login node's small `/tmp` | Set `APPTAINER_TMPDIR`, `APPTAINER_CACHEDIR`, `TMPDIR` to home |
| Build fails with a fakeroot / user namespace error before `%post` | Cluster doesn't allow any fakeroot mode | Build with `sudo` on a machine of the **same architecture** and copy the SIF over |
| `exec format error` or nothing runs | Image or overlay built for another architecture | Rebuild on the target architecture; check with `apptainer sif list` |
| `Permission denied` creating `/opt/venv` | Overlay made without `--create-dir /opt/venv`, or SIF contains a root-owned `/opt/venv` | Recreate the overlay with `--create-dir`; remove venv creation from `%post` |
| `uv venv` fails because the directory exists | Newer uv won't overwrite a non-empty dir | Skip `uv venv`; `uv sync` creates the venv itself |
| `uv sync --frozen` can't find a lockfile | No `uv.lock` for this cluster yet | Run `uv lock --project ...` first |
| Overlay won't mount because it's in use | A writable ext3 overlay can only be open once | Use `:ro` for jobs; only mount read-write when nothing else is using it |
| `torch.cuda.is_available()` → `False` | torch from PyPI (CUDA 13) on a CUDA 12 driver | Check the torch `source` in `uv.lock` points at `download.pytorch.org/whl/cu126`; re-lock |
| JAX cuDNN / cuBLAS version mismatch | Image CUDA libs shadow the pip ones | Comment out `LD_LIBRARY_PATH` and `XLA_FLAGS` in `%environment` (3.7) |
| JAX warning: driver older than ptxas, parallel compilation disabled | Driver 12.2 < ptxas 12.9 from the jax wheels | Harmless; compiles are slower |
| Imports fail for `sbi_compression` | Repo not bound at `/opt/sbi-compression` | Always use `-B $HOME/sbi-compression:/opt/sbi-compression` |
| `Repository not found` + `connect ENOENT .../vscode-git-....sock` | VS Code's GitHub login socket isn't mounted in the container | Run from a VS Code terminal and add `-B /run/user/$(id -u)` (4.4) |
| `Repository not found` + `connect ECONNREFUSED .../vscode-git-....sock` | Terminal points at a dead socket from an old VS Code session | New VS Code terminal, or export `VSCODE_GIT_IPC_HANDLE` to the live socket (4.4) |
| `jaxlib was built using AVX instructions, which your CPU ... do not support` | JAX on the login node (no AVX) | Run on a compute node (4.5) |
| `fatal error: xla/ffi/api/api.h: No such file or directory` building s2fft | JAX failed to import during the build, so the XLA header path was empty | Run `uv sync` on a compute node (4.5) |
| `CUDA_ARCHITECTURES: 52` in the s2fft build | CMake's default overrode s2fft's setting | `--env CUDAARCHS="70;80;89"` (4.5) |
| `configuration disallows users from mounting extfs in setuid mode, try --userns` | Compute nodes disable setuid mounting of `.img` files | Add `--userns` (5.1) |
| `ModuleNotFoundError: No module named 'torch'` with `/opt/venv/bin/python` | `uv sync` failed or stopped, leaving an empty venv | Check the sync log, fix, rerun `uv sync` |
| `nvidia-smi: command not found` / no GPU | On the login node, or `--nv` missing | Use a GPU compute node and pass `--nv` |

---

## 7. Porting this setup to another cluster

1. Check the cluster (section 2): architecture, fakeroot mode, GPU driver's
   CUDA version. Check the login node **and** a compute node separately
   (5.1): `grep -o -w -m1 avx /proc/cpuinfo`,
   `cat /proc/sys/user/max_user_namespaces`, and whether
   `apptainer exec --overlay x.img ...` works with or without `--userns`.
   Also check which GPU models the nodes have, to set `CUDAARCHS` (4.5).
2. Copy the `.def` and change `From:` to a CUDA base image no newer than the
   driver supports.
3. Copy `config_hypatia/` to `config_<cluster>/`. If the driver supports
   CUDA 13, torch from PyPI is fine; otherwise keep a `cu12x` index.
4. Delete any copied `uv.lock` and re-lock on the new cluster.
5. Follow `apptainer-hypatia-step-by-step.md` with the new names and paths.
   Adjust bind mounts for the cluster's storage (e.g. Isambard's `/scratch`
   and `/projects` don't exist on Hypatia).
