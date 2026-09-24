# Setting Up Direct SSH Access Between Clusters (via a Jump Host)

## Goal

Enable a compute cluster (e.g. `login45`/`login40`) to SSH directly into a target
cluster (`hypatia`) through an intermediate jump host (`zuserver`), without
routing anything through a local laptop. This is needed so that long-running
jobs (like a large data transfer) can run unattended in `tmux`/batch, without
depending on the laptop staying connected.

## Topology

```
current cluster (login45)  --SSH-->  zuserver (jump host)  --SSH-->  hypatia (target)
```

- **Current cluster**: e.g. `login45` / `login40` — where jobs are run from.
- **Jump host**: `zuserver1.star.ucl.ac.uk` — an intermediate host you must
  pass through to reach hypatia.
- **Target**: `hypatia-login.hpc.phys.ucl.ac.uk` — final destination.
- Username throughout: `brianycc`.

A separate machine (a personal Mac) already had working SSH access to both
zuserver and hypatia via key-based auth, and was used as a bridge to push the
new key into place on each hop.

## Why this is needed

By default, SSH keys and config live only on the machine you generated them
on. Having key access from your Mac to hypatia does **not** give the compute
cluster access — each machine needs its own keypair authorized on the hops it
needs to reach.

---

## Step 1 — Generate a new SSH keypair on the current cluster

On the current cluster (e.g. `login45`), which had no existing keys at all:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
```

- Creates `~/.ssh/id_ed25519` (private key — never leaves this machine) and
  `~/.ssh/id_ed25519.pub` (public key — safe to copy elsewhere).
- A passphrase can be set for extra security (see Step 5 for how to avoid
  re-entering it constantly).

Print the public key to copy it in later steps:

```bash
cat ~/.ssh/id_ed25519.pub
```

## Step 2 — Authorize the new key on the jump host (zuserver)

This has to be done from a machine that can *already* log into zuserver — in
this case, the Mac (or, once already logged into zuserver via password, the
current cluster itself).

On zuserver:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "ssh-ed25519 AAAA...(paste the pubkey from Step 1)... brianycc.u6pf@login40" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

**Common error hit here:** `~/.ssh/authorized_keys: No such file or
directory` — this means `~/.ssh` doesn't exist yet on that host. Always
`mkdir -p ~/.ssh && chmod 700 ~/.ssh` first.

**Permissions matter**: SSH silently refuses to use `authorized_keys` if
`~/.ssh` or the file itself is group/world-writable. Always `chmod 700
~/.ssh` and `chmod 600 ~/.ssh/authorized_keys`.

## Step 3 — Authorize the same key on the target (hypatia)

Being authorized on zuserver does **not** carry over to hypatia — they are
separate systems (`star.ucl.ac.uk` vs `hpc.phys.ucl.ac.uk`). Repeat the exact
same process while logged into hypatia:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "ssh-ed25519 AAAA...(same pubkey as Step 1)... brianycc.u6pf@login40" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

## Step 4 — Set up SSH config shortcuts on the current cluster

On the current cluster, create/edit `~/.ssh/config`:

```
Host zuserver
    HostName zuserver1.star.ucl.ac.uk
    User brianycc
    IdentityFile ~/.ssh/id_ed25519

Host hypatia
    HostName hypatia-login.hpc.phys.ucl.ac.uk
    User brianycc
    ProxyJump zuserver
    IdentityFile ~/.ssh/id_ed25519
```

`ProxyJump zuserver` tells SSH to automatically tunnel the connection to
`hypatia` through `zuserver`, so a plain `ssh hypatia` does both hops in one
command.

## Step 5 — (Optional) Cache the key passphrase for the session

If the key has a passphrase, each hop will prompt for it separately unless an
agent is running. To avoid retyping it repeatedly in one shell session:

```bash
eval $(ssh-agent)
ssh-add ~/.ssh/id_ed25519
```

Enter the passphrase once; subsequent `ssh` commands in that same shell
session reuse the cached decrypted key.

## Step 6 — Test the full chain

```bash
ssh hypatia
```

First connection to each new host will prompt:
```
The authenticity of host '...' can't be established. ... continue connecting (yes/no/[fingerprint])?
```
Type `yes` (full word — a bare `y` is rejected) to add it to `known_hosts`.
After that, it should log straight into hypatia via zuserver with no password
prompt (aside from the key passphrase, if set and not cached via Step 5).

---

## Troubleshooting notes encountered

- **`ssh: connect to host ... port 22: Connection timed out` / hangs**: means
  the network path between the two clusters is blocked by firewall policy.
  Check with the target cluster's admins whether the source cluster's IP can
  be allowlisted.
- **`Host key verification failed`**: just means the host isn't yet in
  `known_hosts` — accept it with `yes`, or pre-populate via `ssh-keyscan -H
  <host> >> ~/.ssh/known_hosts`.
- **Falls back to asking for a password despite having a key set up**: the
  public key hasn't actually been added to that specific host's
  `authorized_keys` yet (remember: each hop needs it added separately).
- **MFA/Duo**: if a cluster enforces two-factor auth on all logins, key-based
  auth alone may still trigger an interactive prompt, which would block
  unattended/batch jobs. Check with the cluster's support if this becomes an
  issue for automated transfers.

---

## Next step

With `ssh hypatia` working directly from the current cluster (no laptop
dependency), the cluster-to-cluster data transfer (e.g. via `rsync` or `tar`
piped over SSH, run inside `tmux` so it survives disconnects) can be run
directly from the current cluster's login node.
