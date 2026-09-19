"""``aegisq`` command line entry point.

Commands are added incrementally by the project phases; every command lives in
its own ``_cmd_*`` function so the parser stays readable as the surface grows.
"""

from __future__ import annotations

import argparse
import sys

from aegisq import __version__


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report what the local machine can and cannot do."""
    from aegisq.runtime import hardware

    components = hardware.collect()

    if args.json:
        import json

        print(json.dumps(hardware.environment_snapshot(), indent=2, sort_keys=True))
        return 0

    print(f"AegisQ-HPC {__version__}")
    print()
    width = max(len(c.name) for c in components)
    for component in components:
        status = "available" if component.available else "unavailable"
        print(f"  {component.name:<{width}}  {status:<12} {component.detail}")

    extras = {c.name: c.extra for c in components}
    oqs_extra = extras.get("liboqs", {})
    if oqs_extra.get("version"):
        kem = "yes" if oqs_extra.get("ML-KEM-768") else "no"
        sig = "yes" if oqs_extra.get("ML-DSA-65") else "no"
        print()
        print(f"  post-quantum suite: ML-KEM-768 {kem}, ML-DSA-65 {sig}")

    print()
    required_missing = [c.name for c in components if not c.available and c.name == "Native core"]
    if required_missing:
        print("  hint: run `make build` to compile the native core")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegisq",
        description=(
            "AegisQ-HPC: post-quantum secure, communication-aware distributed "
            "quantum circuit simulation."
        ),
    )
    parser.add_argument("--version", action="version", version=f"AegisQ-HPC {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    doctor = subparsers.add_parser(
        "doctor",
        help="report availability of MPI, OpenMP, liboqs, Qiskit and CUDA",
    )
    doctor.add_argument("--json", action="store_true", help="emit a machine-readable snapshot")
    doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
