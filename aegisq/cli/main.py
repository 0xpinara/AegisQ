"""``aegisq`` command line entry point.

Commands are added incrementally by the project phases; every command lives in
its own ``_cmd_*`` function so the parser stays readable as the surface grows.
"""

from __future__ import annotations

import argparse
import sys

from aegisq import __version__


def _load_circuit(source: str, num_qubits: int | None, options: dict[str, str]):
    """Resolve a circuit from a file path or a built-in family name.

    `source` is either a path (`.json` today, `.qasm` once the parser lands)
    or one of the benchmark family names, in which case `--qubits` is
    required.
    """
    import json
    from pathlib import Path

    from aegisq.algorithms import CIRCUIT_FAMILIES, build_circuit
    from aegisq.circuit import Circuit

    path = Path(source)
    if path.exists():
        if path.suffix == ".json":
            return Circuit.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if path.suffix == ".qasm":
            try:
                from aegisq.circuit.qasm import parse_qasm_file
            except ImportError:  # pragma: no cover - only before the parser lands
                raise SystemExit(
                    "this build has no OpenQASM front end; pass a .json circuit instead"
                ) from None
            return parse_qasm_file(path)
        raise SystemExit(f"unsupported circuit file type: {path.suffix}")

    if source not in CIRCUIT_FAMILIES:
        raise SystemExit(
            f"'{source}' is neither an existing file nor a known circuit family "
            f"({', '.join(sorted(CIRCUIT_FAMILIES))})"
        )
    if num_qubits is None:
        raise SystemExit(f"--qubits is required when building the '{source}' family")

    typed: dict[str, object] = {}
    for key, value in options.items():
        try:
            typed[key] = int(value)
        except ValueError:
            typed[key] = value
    return build_circuit(source, num_qubits, **typed)


def _parse_options(items: list[str] | None) -> dict[str, str]:
    """Parse repeated `--option key=value` flags."""
    parsed: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--option expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _cmd_optimize(args: argparse.Namespace) -> int:
    """Search for a communication-minimising qubit placement."""
    import json

    from aegisq.compiler import CommunicationCostModel, StaticCommunicationMapper

    circuit = _load_circuit(args.circuit, args.qubits, _parse_options(args.option))

    try:
        model = CommunicationCostModel(circuit.num_qubits, args.ranks, args.precision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    mapper = StaticCommunicationMapper(model, candidate_budget=args.candidate_budget)
    result = mapper.optimize(circuit)

    if args.json:
        payload = result.as_dict()
        payload["circuit"] = {
            "name": circuit.name,
            "num_qubits": circuit.num_qubits,
            "gates": len(circuit),
            "depth": circuit.depth(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(
        f"Circuit: {circuit.name}  "
        f"({circuit.num_qubits} qubits, {len(circuit)} gates, depth {circuit.depth()})"
    )
    print(
        f"Ranks:   {args.ranks}  precision: {args.precision}  "
        f"shard: {model.shard_amplitudes} amplitudes"
    )
    print()
    print(result.report())
    return 0


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

    optimize = subparsers.add_parser(
        "optimize",
        help="choose a communication-aware qubit placement for a circuit",
        description=(
            "Search the qubit-placement space for the assignment of global "
            "(rank-selecting) qubits that minimises predicted MPI traffic. "
            "Reported figures are cost-model predictions; measured traffic "
            "comes from a benchmark run."
        ),
    )
    optimize.add_argument(
        "circuit",
        help="circuit file (.json, .qasm) or a benchmark family name "
        "(ghz, qft, ising, grover, random)",
    )
    optimize.add_argument("--ranks", type=int, required=True, help="MPI world size (power of two)")
    optimize.add_argument("--qubits", type=int, help="width, when building a named family")
    optimize.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    optimize.add_argument(
        "--option",
        action="append",
        metavar="KEY=VALUE",
        help="extra family option, e.g. --option steps=8",
    )
    optimize.add_argument(
        "--candidate-budget",
        type=int,
        default=200_000,
        help="maximum subsets to score exhaustively before falling back to a heuristic",
    )
    optimize.add_argument("--json", action="store_true", help="emit machine-readable output")
    optimize.set_defaults(func=_cmd_optimize)

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
