# gpusched

Python client for [gpusched](https://github.com/shayonj/gpusched) — the GPU Process Manager.

```bash
pip install git+https://github.com/shayonj/gpusched.git#subdirectory=sdk/python
```

```python
from gpusched import GpuSched

sched = GpuSched()

sched.run("llama", ["python3", "serve.py", "--model", "llama-3-8b"])
sched.freeze("llama")   # GPU freed in ~600ms
sched.thaw("llama")     # restored in ~400ms
sched.swap("llama", "mistral")
```

Zero dependencies — stdlib only. Requires the gpusched daemon to be running. See the [main repo](https://github.com/shayonj/gpusched) for installation.
