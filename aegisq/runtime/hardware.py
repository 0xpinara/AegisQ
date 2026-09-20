"""Environment probing used by ``aegisq doctor`` and by benchmark provenance.

Every probe is defensive: a missing optional component is reported as
unavailable rather than raised, because AegisQ must stay usable on machines
without MPI, liboqs, Qiskit or CUDA.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

_PROBE_TIMEOUT_S = 10.0


def _run(cmd: list[str]) -> str | None:
    """Run a probe command and return stdout, or None when it is unusable."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or out.stderr).strip()


@dataclass(frozen=True)
class Component:
    """Availability record for one optional dependency."""

    name: str
    available: bool
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def python_info() -> Component:
    return Component(
        name="Python",
        available=True,
        detail=f"{platform.python_version()} ({sys.executable})",
        extra={"version": platform.python_version(), "executable": sys.executable},
    )


def native_core_info() -> Component:
    from aegisq import native_core, native_import_error

    core = native_core()
    if core is None:
        err = native_import_error()
        return Component(
            name="Native core",
            available=False,
            detail=f"not built ({type(err).__name__ if err else 'ImportError'})",
        )
    info = core.build_info()
    return Component(
        name="Native core",
        available=True,
        detail=f"{info['version']} built with {info['compiler']}",
        extra=dict(info),
    )


def openmp_info() -> Component:
    from aegisq import native_core

    core = native_core()
    if core is None:
        return Component("OpenMP", False, "native core not built")
    if not core.has_openmp():
        return Component("OpenMP", False, "core compiled without OpenMP")
    threads = core.max_threads()
    return Component("OpenMP", True, f"{threads} thread(s)", {"max_threads": threads})


def mpi_info() -> Component:
    """Detect an MPI launcher and, if built, MPI support inside the core."""
    from aegisq import native_core

    core = native_core()
    core_mpi = bool(core.has_mpi()) if core is not None else False

    launcher = None
    version_line = None
    for candidate in ("mpirun", "mpiexec", "srun"):
        if shutil.which(candidate):
            launcher = candidate
            break
    if launcher is not None:
        raw = _run([launcher, "--version"])
        if raw:
            version_line = raw.splitlines()[0].strip()

    # MPI_Get_library_version may be called before MPI_Init, so this does not
    # start a communicator just to answer a diagnostic question.
    library = core.mpi_library_version() if core is not None and core_mpi else None

    available = core_mpi and launcher is not None
    if launcher is None:
        detail = "no launcher found (mpirun/mpiexec/srun)"
    elif not core_mpi:
        detail = f"{version_line or launcher} present, but core built without MPI"
    else:
        detail = version_line or launcher
    return Component(
        "MPI",
        available,
        detail,
        {
            "launcher": launcher,
            "version": version_line,
            "core_mpi": core_mpi,
            "library": library,
        },
    )


def cpu_info() -> Component:
    model = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        model = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or model
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    logical = os.cpu_count() or 1
    return Component(
        "CPU",
        True,
        f"{model} ({logical} logical cores)",
        {"model": model, "logical_cores": logical, "machine": platform.machine()},
    )


def liboqs_info() -> Component:
    try:
        import oqs  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on local install
        return Component("liboqs", False, f"import failed: {type(exc).__name__}")
    try:
        kems = set(oqs.get_enabled_kem_mechanisms())
        sigs = set(oqs.get_enabled_sig_mechanisms())
        lib_version = oqs.oqs_version()
    except Exception as exc:  # pragma: no cover
        return Component("liboqs", False, f"probe failed: {exc}")
    return Component(
        "liboqs",
        True,
        f"{lib_version}",
        {
            "version": lib_version,
            "ML-KEM-768": "ML-KEM-768" in kems,
            "ML-DSA-65": "ML-DSA-65" in sigs,
            "kems": sorted(k for k in kems if k.startswith("ML-KEM")),
            "sigs": sorted(s for s in sigs if s.startswith("ML-DSA")),
        },
    )


def qiskit_info() -> Component:
    try:
        import qiskit  # type: ignore[import-not-found]
    except Exception:
        return Component("Qiskit (validation only)", False, "not installed")
    return Component(
        "Qiskit (validation only)",
        True,
        qiskit.__version__,
        {"version": qiskit.__version__},
    )


def cuda_info() -> Component:
    raw = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if not raw:
        return Component("CUDA", False, "no CUDA device detected")
    first = raw.splitlines()[0].strip()
    return Component("CUDA", True, first, {"devices": raw.splitlines()})


def compiler_info() -> Component:
    from aegisq import native_core

    core = native_core()
    if core is None:
        return Component("C++ compiler", False, "native core not built")
    return Component("C++ compiler", True, core.compiler(), {"compiler": core.compiler()})


def system_memory_bytes() -> int | None:
    """Physical RAM of this machine, or None when it cannot be determined."""
    if sys.platform == "darwin":
        raw = _run(["sysctl", "-n", "hw.memsize"])
        return int(raw) if raw and raw.isdigit() else None
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            return None
    return None


#: Bytes per complex amplitude, by precision.
AMPLITUDE_BYTES = {"fp64": 16, "fp32": 8}


def memory_estimate(num_qubits: int, ranks: int = 1, precision: str = "fp64") -> dict[str, Any]:
    """Memory a distributed state vector will occupy.

    This is arithmetic, not a measurement, and it is reported as such. The
    peak figure matters more than the shard size: the runtime also holds an
    incoming-shard buffer (up to one shard) and a packing buffer (up to half a
    shard) for the gate placements that exchange data, so a rank's working set
    can reach 2.5x the shard itself.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")
    if ranks < 1 or ranks & (ranks - 1):
        raise ValueError(f"rank count must be a power of two, got {ranks}")
    if precision not in AMPLITUDE_BYTES:
        raise ValueError(f"unknown precision {precision!r}; use 'fp64' or 'fp32'")

    global_qubits = ranks.bit_length() - 1
    local_qubits = num_qubits - global_qubits
    if local_qubits < 1:
        raise ValueError(
            f"{num_qubits} qubits over {ranks} ranks leaves no local qubits; "
            f"use at most {2 ** (num_qubits - 1)} ranks"
        )

    amplitude_bytes = AMPLITUDE_BYTES[precision]
    total_amplitudes = 1 << num_qubits
    shard_amplitudes = 1 << local_qubits
    shard_bytes = shard_amplitudes * amplitude_bytes

    system_memory = system_memory_bytes()
    peak_per_rank = shard_bytes * 5 // 2

    return {
        "num_qubits": num_qubits,
        "ranks": ranks,
        "precision": precision,
        "amplitude_bytes": amplitude_bytes,
        "local_qubits": local_qubits,
        "global_qubits": global_qubits,
        "total_amplitudes": total_amplitudes,
        "total_bytes": total_amplitudes * amplitude_bytes,
        "amplitudes_per_rank": shard_amplitudes,
        "bytes_per_rank": shard_bytes,
        "peak_bytes_per_rank": peak_per_rank,
        "exchange_buffer_bytes": shard_bytes,
        "packing_buffer_bytes": shard_bytes // 2,
        "system_memory_bytes": system_memory,
        "fits_on_this_host": (
            None if system_memory is None else peak_per_rank * ranks <= system_memory
        ),
    }


def format_bytes(value: int) -> str:
    for unit, scale in (("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= scale:
            return f"{value / scale:.2f} {unit}"
    return f"{value} B"


def collect() -> list[Component]:
    """Full diagnostic sweep, in display order."""
    return [
        python_info(),
        native_core_info(),
        compiler_info(),
        cpu_info(),
        openmp_info(),
        mpi_info(),
        liboqs_info(),
        qiskit_info(),
        cuda_info(),
    ]


def environment_snapshot() -> dict[str, Any]:
    """Machine-readable environment record embedded in benchmark provenance."""
    components = {
        c.name: {"available": c.available, "detail": c.detail, **c.extra} for c in collect()
    }
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "components": components,
    }
