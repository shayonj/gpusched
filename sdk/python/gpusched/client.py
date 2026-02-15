"""Core client for communicating with the gpusched daemon."""

from __future__ import annotations

import json
import socket
import time
from typing import Any

DEFAULT_SOCKET = "/tmp/gpusched.sock"


class GpuSched:
    """Client for the gpusched GPU Process Manager.

    Communicates with the daemon over a Unix domain socket using
    a JSON-lines protocol. No external dependencies required.

    Example::

        from gpusched import GpuSched

        sched = GpuSched()
        sched.run("model-a", ["python3", "serve.py", "--model", "llama-3"])
        sched.freeze("model-a")
        sched.thaw("model-a")
    """

    def __init__(self, socket_path: str = DEFAULT_SOCKET):
        self.socket_path = socket_path

    def _call(self, method: str, params: Any = None) -> dict:
        """Send a request and return the result dict."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.socket_path)
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            raise ConnectionError(
                "gpusched daemon is not running.\n"
                "  Start it with: sudo gpusched daemon\n"
                "  Or via systemd: sudo systemctl start gpusched"
            ) from exc

        try:
            req: dict[str, Any] = {"method": method}
            if params is not None:
                req["params"] = params
            sock.sendall(json.dumps(req).encode() + b"\n")

            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk

            resp = json.loads(buf.decode().strip())
            if not resp.get("ok"):
                raise RuntimeError(resp.get("error", "unknown error"))
            return resp.get("result")
        finally:
            sock.close()

    def run(self, name: str, cmd: list[str], gpu: int = 0) -> dict:
        """Spawn a managed GPU process."""
        return self._call("run", {"name": name, "cmd": cmd, "gpu": gpu})

    def freeze(self, name: str) -> dict:
        """Checkpoint a process from GPU to host RAM."""
        return self._call("freeze", {"name": name})

    def thaw(self, name: str) -> dict:
        """Restore a frozen process back to the GPU."""
        return self._call("thaw", {"name": name})

    def kill(self, name: str) -> dict:
        """Terminate a managed process."""
        return self._call("kill", {"name": name})

    def status(self) -> dict:
        """Return full system state."""
        return self._call("status")

    def logs(self, name: str, lines: int = 50) -> dict:
        """Return recent stdout/stderr for a process."""
        return self._call("logs", {"name": name, "lines": lines})

    def swap(self, off: str, on: str) -> tuple[dict, dict]:
        """Freeze *off*, then thaw *on*."""
        fr = self.freeze(off)
        th = self.thaw(on)
        return fr, th

    def processes(self) -> list[dict]:
        """Return the list of managed processes."""
        return self.status().get("processes", [])

    def gpu_free_mb(self, gpu: int = 0) -> int:
        """Return free GPU memory in MB."""
        for g in self.status().get("gpus", []):
            if g["index"] == gpu:
                return g.get("mem_free_mb", 0)
        return 0

    def wait_ready(
        self,
        name: str,
        timeout: float = 120.0,
        poll: float = 2.0,
    ) -> dict:
        """Block until *name* reports GPU memory usage (model loaded)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for p in self.processes():
                if p["name"] == name and p.get("mem_mb", 0) > 0:
                    return p
            time.sleep(poll)
        raise TimeoutError(f"{name} did not load within {timeout}s")
