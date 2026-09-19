"""AegisQ-HPC: post-quantum secure, communication-aware distributed quantum simulation.

The package is organised in layers:

``aegisq.circuit``
    Circuit/gate data structures and the OpenQASM subset front end.
``aegisq.runtime``
    Reference (NumPy) and native (C++/MPI) execution backends.
``aegisq.compiler``
    Communication cost model and qubit-placement optimisation.
``aegisq.secure``
    ML-KEM / ML-DSA protected job envelopes.
``aegisq.provenance``
    Hashing, Merkle trees and signed execution manifests.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__", "native_core", "has_native_core"]

_NATIVE_IMPORT_ERROR: Exception | None = None


def native_core():
    """Return the compiled ``_aegisq_core`` extension module, or ``None``.

    The native core is optional: the pure-Python reference backend works
    without it, which keeps the package importable before ``make build``.
    """
    global _NATIVE_IMPORT_ERROR
    try:
        from . import _aegisq_core  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - exercised on unbuilt trees
        _NATIVE_IMPORT_ERROR = exc
        return None
    return _aegisq_core


def has_native_core() -> bool:
    """True when the compiled extension is importable."""
    return native_core() is not None


def native_import_error() -> Exception | None:
    """The last import failure of the native core, for diagnostics."""
    native_core()
    return _NATIVE_IMPORT_ERROR
