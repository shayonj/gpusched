# gpusched

```
 ██████╗ ██████╗ ██╗   ██╗███████╗ ██████╗██╗  ██╗███████╗██████╗
██╔════╝ ██╔══██╗██║   ██║██╔════╝██╔════╝██║  ██║██╔════╝██╔══██╗
██║  ███╗██████╔╝██║   ██║███████╗██║     ███████║█████╗  ██║  ██║
██║   ██║██╔═══╝ ██║   ██║╚════██║██║     ██╔══██║██╔══╝  ██║  ██║
╚██████╔╝██║     ╚██████╔╝███████║╚██████╗██║  ██║███████╗██████╔╝
 ╚═════╝ ╚═╝      ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═════╝
```

**Freeze and restore GPU processes in milliseconds.**

---

Loading an LLM takes 15–30 seconds. Restoring a frozen one takes under a second.

`gpusched` wraps NVIDIA's `cuda-checkpoint` into a process manager. You freeze a GPU process, its VRAM gets parked in host RAM, and the GPU is free. You thaw it, the model is back.

## Install

```bash
curl -sSL https://raw.githubusercontent.com/shayonj/gpusched/main/install.sh | sudo bash
```

Or pin a version:

```bash
curl -sSL https://raw.githubusercontent.com/shayonj/gpusched/main/install.sh | GPUSCHED_VERSION=0.1.0 sudo -E bash
```

Downloads the binary from GitHub Releases. Also installs `cuda-checkpoint` and a systemd service. Requires Linux, NVIDIA driver 580+.

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

Terminal UI with live GPU/RAM utilization, process table, event log. Keyboard driven: `f` freeze, `t` thaw, `x` kill, `q` quit.

## How It Works

```
                freeze (~600ms)
  GPU VRAM ─────────────────▶ Host RAM
     ◀──────────────────────
                thaw (~400ms)
```

When you freeze, gpusched calls `cuda-checkpoint` to snapshot GPU state into host RAM, then stops the process with `SIGSTOP`. When you thaw, it restores the snapshot and resumes with `SIGCONT`. The process never knows it was paused.

On multi-GPU machines, `gpusched migrate` can move a process from one GPU to another by checkpointing on the source and restoring on the target.

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
gpusched migrate NAME --to GPU                 Move to a different GPU
```

## Advanced

### Daemon

```bash
sudo gpusched daemon --ram-budget 80G
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

- Single machine only. No multi-node coordination.
- Requires root (or `CAP_SYS_ADMIN`) for `cuda-checkpoint`.
- Snapshots aren't portable across GPU architectures.
- Frozen processes live in host RAM — you need enough free host memory to hold the GPU snapshot.
- No HTTP API — the daemon only speaks Unix socket today.
- `cuda-checkpoint` does not support UVM or IPC memory ([upstream limitation](https://github.com/NVIDIA/cuda-checkpoint#functionality)).

## Future Exploration Ideas

- **Disk-backed snapshots.** Today frozen processes live in host RAM only. A disk tier would allow unlimited frozen models and survive reboots. This is blocked on NVIDIA's `cuda-checkpoint` adding direct GPU-to-file checkpointing ([cuda-checkpoint#33](https://github.com/NVIDIA/cuda-checkpoint/issues/33)). CRIU-based dump/restore does not currently work for PyTorch processes.
- **HTTP API on the daemon.** Would make gpusched remotely controllable and open the door to language-agnostic clients, Prometheus metrics, and integration with existing orchestration tools.
- **Policy-based eviction.** Priority levels, per-process TTLs, auto-freeze on idle.

## License

Apache 2.0
