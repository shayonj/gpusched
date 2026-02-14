"""Integration tests — require a running gpusched daemon.

Run with: pytest -m integration
"""

import os
import time

import pytest

from gpusched import GpuSched

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _skip_if_no_daemon():
    if not os.path.exists("/tmp/gpusched.sock"):
        pytest.skip("gpusched daemon not running")


def test_status_live():
    sched = GpuSched()
    result = sched.status()
    assert "gpus" in result
    assert "capabilities" in result


def test_run_freeze_thaw_kill():
    sched = GpuSched()
    result = sched.run("pytest-test", ["sleep", "3600"])
    assert result["pid"] > 0
    time.sleep(1)
    try:
        sched.freeze("pytest-test")
        sched.thaw("pytest-test")
    except RuntimeError:
        pass
    sched.kill("pytest-test")
