"""Unit tests for the gpusched Python client."""

import json
import os
import socket
import threading

import pytest

from gpusched import GpuSched


class MockDaemon:
    """Minimal fake gpusched daemon for unit testing."""

    def __init__(self, sock_path: str):
        self.sock_path = sock_path
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(sock_path)
        self.server.listen(5)
        self.server.settimeout(5)
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._responses: dict[str, dict] = {}

    def set_response(self, method: str, result: dict):
        self._responses[method] = result

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        self.server.close()

    def _serve(self):
        while self._running:
            try:
                conn, _ = self.server.accept()
            except (OSError, socket.timeout):
                break
            try:
                data = b""
                while b"\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                req = json.loads(data.decode().strip())
                method = req.get("method", "")
                if method in self._responses:
                    resp = {"ok": True, "result": self._responses[method]}
                else:
                    resp = {"ok": False, "error": f"unknown method: {method}"}
                conn.sendall(json.dumps(resp).encode() + b"\n")
            finally:
                conn.close()


@pytest.fixture
def daemon(tmp_path):
    sock_path = f"/tmp/gpusched_test_{os.getpid()}.sock"
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    d = MockDaemon(sock_path)
    d.set_response("status", {
        "gpus": [{"index": 0, "name": "Test GPU", "mem_total_mb": 80000, "mem_used_mb": 5000, "mem_free_mb": 75000}],
        "processes": [
            {"name": "model-a", "pid": 100, "state": "active", "gpu": 0, "mem_mb": 1500, "age": "10s", "tier": "gpu"},
        ],
        "memory": {"snapshots_mb": 0, "host_ram_budget_mb": 100000},
        "metrics": {"requests": 5, "freezes": 2, "thaws": 1, "avg_freeze_ms": 600, "avg_thaw_ms": 400},
        "capabilities": {"cuda_checkpoint": True, "driver_version": "580.126.09"},
    })
    d.set_response("run", {"name": "test-proc", "pid": 999})
    d.set_response("freeze", {"name": "model-a", "duration_ms": 609, "mem_mb": 1500})
    d.set_response("thaw", {"name": "model-a", "duration_ms": 427, "mem_mb": 1500})
    d.set_response("kill", {"name": "model-a"})
    d.set_response("logs", {"name": "model-a", "output": "Loading model...\nReady.", "lines": 2})
    d.start()
    yield d, sock_path
    d.stop()
    if os.path.exists(sock_path):
        os.unlink(sock_path)


class TestGpuSchedClient:

    def test_status(self, daemon):
        _, sock = daemon
        sched = GpuSched(socket_path=sock)
        result = sched.status()
        assert len(result["gpus"]) == 1
        assert result["gpus"][0]["name"] == "Test GPU"
        assert len(result["processes"]) == 1
        assert result["processes"][0]["name"] == "model-a"

    def test_run(self, daemon):
        _, sock = daemon
        sched = GpuSched(socket_path=sock)
        result = sched.run("test-proc", ["python3", "serve.py"])
        assert result["name"] == "test-proc"
        assert result["pid"] == 999

    def test_freeze(self, daemon):
        _, sock = daemon
        sched = GpuSched(socket_path=sock)
        result = sched.freeze("model-a")
        assert result["duration_ms"] == 609

    def test_thaw(self, daemon):
        _, sock = daemon
        sched = GpuSched(socket_path=sock)
        result = sched.thaw("model-a")
        assert result["duration_ms"] == 427

    def test_kill(self, daemon):
        _, sock = daemon
        sched = GpuSched(socket_path=sock)
        result = sched.kill("model-a")
        assert result["name"] == "model-a"

    def test_logs(self, daemon):
        _, sock = daemon
        sched = GpuSched(socket_path=sock)
        result = sched.logs("model-a", lines=2)
        assert "Ready." in result["output"]

    def test_swap(self, daemon):
        _, sock = daemon
        sched = GpuSched(socket_path=sock)
        fr, th = sched.swap("model-a", "model-a")
        assert fr["duration_ms"] == 609
        assert th["duration_ms"] == 427

    def test_processes(self, daemon):
        _, sock = daemon
        sched = GpuSched(socket_path=sock)
        procs = sched.processes()
        assert len(procs) == 1
        assert procs[0]["state"] == "active"

    def test_gpu_free_mb(self, daemon):
        _, sock = daemon
        sched = GpuSched(socket_path=sock)
        assert sched.gpu_free_mb(0) == 75000

    def test_connection_error(self):
        sched = GpuSched(socket_path="/tmp/gpusched_nonexistent_test.sock")
        with pytest.raises(ConnectionError, match="daemon is not running"):
            sched.status()

    def test_server_error(self, daemon):
        _, sock = daemon
        sched = GpuSched(socket_path=sock)
        with pytest.raises(RuntimeError, match="unknown method"):
            sched._call("nonexistent_method")
