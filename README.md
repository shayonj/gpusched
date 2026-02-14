# gpusched

```
 ██████╗ ██████╗ ██╗   ██╗███████╗ ██████╗██╗  ██╗███████╗██████╗
██╔════╝ ██╔══██╗██║   ██║██╔════╝██╔════╝██║  ██║██╔════╝██╔══██╗
██║  ███╗██████╔╝██║   ██║███████╗██║     ███████║█████╗  ██║  ██║
██║   ██║██╔═══╝ ██║   ██║╚════██║██║     ██╔══██║██╔══╝  ██║  ██║
╚██████╔╝██║     ╚██████╔╝███████║╚██████╗██║  ██║███████╗██████╔╝
 ╚═════╝ ╚═╝      ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═════╝
```

**Easily Freeze and restore GPU processes in milliseconds.**

---

Loading an LLM takes 15–30 seconds. Restoring a frozen one takes under a second.

`gpusched` is an exploratory project that wraps NVIDIA's `cuda-checkpoint` into a process manager. You freeze a GPU process, its VRAM gets parked in host RAM, and the GPU is free. When RAM is tight, the daemon can evict older snapshots to disk via CRIU. You thaw it, the model is back.

## Install

```bash
curl -sSL https://raw.githubusercontent.com/shayonj/gpusched/main/install.sh | sudo bash
```

Or pin a version:

```bash
curl -sSL https://raw.githubusercontent.com/shayonj/gpusched/main/install.sh | GPUSCHED_VERSION=0.1.0 sudo -E bash
```

Downloads the binary from GitHub Releases. Also installs `cuda-checkpoint`, CRIU, and a systemd service. Requires Linux, NVIDIA driver 580+.

## Quick Start

```bash
# Start two models
gpusched run --name llama-inf   -- python3 serve.py --model llama-3-8b
gpusched run --name mistral-inf -- python3 serve.py --model mistral-7b

# See what's running
gpusched status

# Freeze llama (frees GPU in ~600ms)
gpusched freeze llama-inf

gpusched freeze llama-inf && gpusched thaw mistral-inf   # swap
```

## Python SDK

```bash
pip install git+https://github.com/shayonj/gpusched.git#subdirectory=sdk/python
```

```python
from gpusched import GpuSched

sched = GpuSched()
sched.run("policy", ["python3", "serve.py", "--model", "llama-3-8b"])
sched.run("reward", ["python3", "serve.py", "--model", "reward-7b"])

for step in range(100):
    sched.freeze("reward")
    sched.thaw("policy")            # generate rollouts
    sched.swap("policy", "reward")  # score rewards
    sched.swap("reward", "policy")  # update policy
```

Zero dependencies. Stdlib `socket` + `json`. Talks to the daemon over a Unix socket. See [`sdk/python/`](sdk/python/) for the full API.

## Dashboard

```bash
gpusched dashboard
```

Terminal UI with live GPU/RAM/disk utilization, process table, event log. Keyboard driven: `f` freeze, `t` thaw, `x` kill, `q` quit.

## How It Works

Three memory tiers. Processes move between them:

```
                freeze (~600ms)              evict (RAM full)
  GPU VRAM ─────────────────▶ Host RAM ─────────────────────▶ NVMe Disk
     ◀──────────────────────      ◀──────────────────────────
                thaw (~400ms)                thaw (~6s)
```

When you freeze, gpusched calls `cuda-checkpoint` to snapshot GPU state into host RAM, then stops the process with `SIGSTOP`. When you thaw, it restores the snapshot and resumes with `SIGCONT`. The process never knows it was paused.

If host RAM fills up, the daemon evicts the least recently used snapshot to disk via CRIU. Thawing from disk is slower (~6s) but the data survives reboots.

## Benchmarks

H100 PCIe, driver 580.126.09:

| Model        | GPU Memory | Freeze   | Thaw   | Cold Start |
| ------------ | ---------- | -------- | ------ | ---------- |
| Qwen2.5-0.5B | 1,442 MB   | 609 ms   | 427 ms | ~15s       |
| Qwen2.5-1.5B | 3,584 MB   | 1,319 ms | 832 ms | ~25s       |

Freeze + thaw is 25–30x faster than loading from scratch.

## CLI

```
gpusched daemon                                Start the daemon (root)
gpusched run --name NAME -- CMD [ARGS...]      Spawn a managed process
gpusched freeze NAME                           Checkpoint → host RAM
gpusched thaw NAME                             Restore → GPU
gpusched kill NAME                             Terminate
gpusched status [--json]                       Processes + GPU state
gpusched logs NAME [-n LINES]                  Process stdout/stderr
gpusched dashboard                             Interactive TUI
gpusched hibernate NAME                        Snapshot → disk (CRIU)
gpusched fork NAME --copies N                  Clone into N processes
gpusched migrate NAME --to GPU                 Move to a different GPU
```

## Advanced

### Daemon

```bash
sudo gpusched daemon \
  --ram-budget 80G \
  --disk-budget 500G \
  --disk-dir /var/lib/gpusched/snapshots
```

```bash
sudo systemctl status gpusched
sudo journalctl -u gpusched -f
```

### Wire Protocol

JSON-lines over `/tmp/gpusched.sock`. The Python SDK uses this, but anything can:

```bash
echo '{"method":"freeze","params":{"name":"train"}}' | socat - UNIX-CONNECT:/tmp/gpusched.sock
```

## Development

```bash
make build            # build for current platform
make build-linux      # cross-compile for linux/amd64
make test             # go + python tests
sudo make install     # install to /usr/local/bin
```

## Limitations

- Single machine only. No multi-node coordination yet.
- Requires root (or `CAP_SYS_ADMIN`) for `cuda-checkpoint`.
- Snapshots aren't portable across GPU architectures.
- CRIU restores can hit PID conflicts. No PID namespace isolation yet.
- Eviction under RAM pressure is LRU-only — no priority or policy controls.
- No HTTP API — the daemon only speaks Unix socket today.

## Future Exploration

Some directions I'm thinking about:

- **HTTP API on the daemon.** Would make gpusched remotely controllable and open the door to language-agnostic clients, Prometheus metrics, and integration with existing orchestration tools.
- **K8s integration.** Run the daemon as a DaemonSet on GPU nodes. Training scripts and serving pods talk to it via the SDK. Let K8s handle scheduling, let gpusched handle the freeze/thaw lifecycle.
- **Multi-node snapshot transfer.** Freeze on node A, ship the checkpoint over the network, restore on node B. Same GPU architecture required, but still faster than a cold start for large models.
- **Policy-based eviction.** Priority levels, per-process TTLs, auto-freeze on idle. Right now eviction is pure LRU — real workloads need more control over what gets evicted and when.
- **Crash recovery.** If the daemon restarts, reconnect to managed processes that are still alive. Today a daemon restart loses track of everything.

## License

Apache 2.0
