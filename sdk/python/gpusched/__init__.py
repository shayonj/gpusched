"""gpusched — Python client for the gpusched GPU Process Manager.

    pip install git+https://github.com/shayonj/gpusched.git#subdirectory=sdk/python

    from gpusched import GpuSched

    sched = GpuSched()
    sched.run("model-a", ["python3", "serve.py"])
    sched.freeze("model-a")
    sched.thaw("model-a")
"""

from gpusched.client import GpuSched

__version__ = "0.1.0"
__all__ = ["GpuSched"]
