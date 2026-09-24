# Methods for Transferring Files Between Clusters

Companion to `ssh-cluster-access-setup.md`, which covers getting passwordless
SSH working between clusters. This file covers the actual data transfer once
that access is in place.

## Prerequisite: unlock the SSH key once via `ssh-agent`

Before running any transfer — especially a parallel one — cache the key's
passphrase for the shell session so every connection reuses it instead of
prompting separately:

```bash
eval $(ssh-agent)
ssh-add ~/.ssh/id_ed25519
```

**Why this matters for parallel transfers:** if the agent isn't running and
several SSH connections open at once, each prompts for the passphrase on the
same terminal simultaneously. The prompts collide, keystrokes go to the wrong
prompt, key auth fails, and SSH falls back to password auth — which then
fails too if password login isn't set up for that host. Running `ssh-add`
first avoids this entirely.

## Method 1 — Single `rsync` transfer

For one directory/file at a time:

```bash
rsync -avP --partial --append-verify \
  /local/path/to/source \
  hypatia:/remote/destination/
```

- `-a` — archive mode (preserves permissions, timestamps, symlinks, etc.)
- `-v` — verbose
- `-P` — show progress and keep partially-transferred files
- `--partial` — keep partial files if interrupted, so a resume doesn't restart from zero
- `--append-verify` — resume an interrupted large-file transfer by appending, then checksum-verify at the end

Good default for a single transfer that might get interrupted (SSH drop,
walltime limit, etc.) and needs to resume cleanly.

## Method 2 — Parallel `rsync` transfers via `xargs -P`

For transferring several independent directories at once, to make use of
available bandwidth:

```bash
printf '%s\n' \
  /path/to/dir1 \
  /path/to/dir2 \
  /path/to/dir3 \
  /path/to/dir4 \
| xargs -P 4 -I{} rsync -avP --partial --append-verify {} hypatia:/remote/destination/
```

- `xargs -P 4` launches up to 4 rsync processes concurrently, one per input line.
- Each gets its own SSH connection, so make sure `ssh-agent` is unlocked first (see above).
- Pick `-P N` based on available network bandwidth and how many concurrent
  connections the remote host's SSH daemon allows — too high can trigger
  `MaxStartups` limits on the server.

### Verifying transfers are actually running in parallel

```bash
pgrep -af rsync          # lists each rsync process; multiple entries with
                          # different source paths running at once = parallel
ss -tnp | grep ssh | wc -l   # count of live SSH connections; should match -P N
```

Interleaved progress output (file names from different source directories
appearing in the same stream, each with independent `xfr#` counters) is also
a reliable visual sign — sequential transfers would finish one directory
entirely before ever printing the next directory's name.

## Method 3 — `tar` piped over SSH (for many small files)

`rsync` has per-file overhead that adds up with large numbers of small files.
Streaming a tar archive avoids that:

```bash
tar -cf - /local/path/to/source | ssh hypatia 'tar -xf - -C /remote/destination/'
```

Trade-off: no resume support if the connection drops partway — best for
transfers that reliably complete in one go, or as a manual re-run if
interrupted.

## Running unattended (surviving disconnects)

Long transfers should run inside `tmux` so they survive a dropped
connection or closed laptop lid:

```bash
tmux new -s transfer
# run the rsync/tar command inside this session
# detach with: Ctrl-b, then d
# reattach later with: tmux attach -t transfer
```

## Quick decision guide

| Situation | Method |
|---|---|
| One directory/file, might get interrupted | Method 1 (single rsync) |
| Several independent directories, want to use full bandwidth | Method 2 (parallel rsync) |
| Many small files, transfer expected to complete in one go | Method 3 (tar over ssh) |
| Any transfer expected to run longer than you'll stay connected | Wrap in `tmux` |
