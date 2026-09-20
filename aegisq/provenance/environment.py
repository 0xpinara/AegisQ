"""Capture the environment a result was produced in.

A result without its environment is not reproducible and not auditable. The
snapshot records the machine, the toolchain and the exact code revision —
including whether the working tree was dirty, because attributing a
measurement to a commit that does not contain the code that produced it is
worse than admitting the uncertainty.
"""

from __future__ import annotations

import platform
import socket
from typing import Any

from aegisq import __version__


def capture() -> dict[str, Any]:
    """Environment record embedded in signed result manifests."""
    from aegisq import native_core
    from aegisq.benchmark.runner import git_commit
    from aegisq.runtime import hardware
    from aegisq.runtime.distributed import mpi_library_version, world_size

    core = native_core()
    commit, dirty = git_commit()
    cpu = hardware.cpu_info()

    return {
        "aegisq_version": __version__,
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "compiler": core.compiler() if core is not None else "none",
        "openmp": bool(core.has_openmp()) if core is not None else False,
        "threads_per_rank": int(core.max_threads()) if core is not None else 1,
        "mpi_library": mpi_library_version(),
        "world_size": world_size(),
        "cpu_model": cpu.extra.get("model", "unknown"),
        "logical_cores": cpu.extra.get("logical_cores", 0),
    }
