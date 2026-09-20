"""``aegisq`` command line entry point.

Commands are added incrementally by the project phases; every command lives in
its own ``_cmd_*`` function so the parser stays readable as the surface grows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a circuit, distributing it when launched under mpirun."""
    import json

    from aegisq.runtime import Simulator
    from aegisq.runtime.distributed import is_distributed, preferred_backend, rank, world_size

    circuit = _load_circuit(args.circuit, args.qubits, _parse_options(args.option))
    backend = args.backend if args.backend != "auto" else preferred_backend()
    is_lead = rank() == 0

    options: dict[str, object] = {}
    mapping_report = None
    if args.optimize:
        if not is_distributed():
            if is_lead:
                print("note: --optimize has no effect on a single rank", file=sys.stderr)
        else:
            from aegisq.compiler import optimize_placement

            mapping_report = optimize_placement(circuit, world_size(), args.precision)
            options["mapping"] = list(mapping_report.mapping)

    simulator = Simulator(backend, precision=args.precision, **options)
    result = simulator.run(
        circuit,
        shots=args.shots,
        seed=args.seed,
        save_statevector=args.statevector,
    )

    if not is_lead:
        return 0

    if args.json:
        payload = {
            "circuit": {
                "name": circuit.name,
                "num_qubits": circuit.num_qubits,
                "gates": len(circuit),
                "depth": circuit.depth(),
            },
            "backend": backend,
            "precision": args.precision,
            "shots": args.shots,
            "seed": args.seed,
            "counts": result.counts,
            "metrics": {k: v for k, v in result.metrics.items() if k != "per_opcode"},
        }
        if mapping_report is not None:
            payload["placement"] = mapping_report.as_dict()
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0

    print(
        f"Circuit: {circuit.name}  "
        f"({circuit.num_qubits} qubits, {len(circuit)} gates, depth {circuit.depth()})"
    )
    print(f"Backend: {backend}  precision: {args.precision}  ranks: {world_size()}")
    if mapping_report is not None:
        print(
            f"Placement: global qubits {list(mapping_report.global_qubits)} "
            f"(predicted {mapping_report.reduction * 100:.1f}% less traffic)"
        )
    print()

    if result.counts:
        total = sum(result.counts.values())
        print(f"Counts ({total} shots, top {args.top}):")
        for bitstring, count in result.most_frequent(args.top):
            bar = "#" * max(1, round(40 * count / total))
            print(f"  {bitstring}  {count:>8}  {count / total * 100:6.2f}%  {bar}")
        print()

    if args.statevector and result.statevector is not None:
        import numpy as np

        state = np.asarray(result.statevector)
        print("Non-zero amplitudes:")
        for index in np.flatnonzero(np.abs(state) > 1e-9)[: args.top]:
            print(f"  |{index:0{circuit.num_qubits}b}>  {state[index]:.6f}")
        print()

    metrics = result.metrics
    print("Metrics:")
    print(f"  wall time:            {metrics.get('wall_seconds', 0.0) * 1000:.2f} ms")
    if "bytes_sent" in metrics:
        print(
            f"  MPI bytes sent:       {metrics['bytes_sent']} "
            f"({metrics['bytes_sent'] / 2**20:.2f} MiB, summed over ranks)"
        )
        print(f"  pairwise exchanges:   {metrics['pairwise_exchanges']}")
        print(
            f"  communication time:   {metrics['communication_seconds'] * 1000:.2f} ms "
            f"(slowest rank)"
        )
        print(f"  compute time:         {metrics['compute_seconds'] * 1000:.2f} ms (slowest rank)")
    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    """Report the memory a distributed state vector would occupy."""
    import json

    from aegisq.runtime.hardware import format_bytes, memory_estimate

    try:
        estimate = memory_estimate(args.qubits, args.ranks, args.precision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    if args.json:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0

    print(
        f"State vector: {args.qubits} qubits, {args.precision} "
        f"({estimate['amplitude_bytes']} bytes per amplitude)"
    )
    print()
    print(f"  total amplitudes:      {estimate['total_amplitudes']:,}")
    print(f"  total state memory:    {format_bytes(estimate['total_bytes'])}")
    print(f"  ranks:                 {estimate['ranks']}")
    print(f"  local qubits:          {estimate['local_qubits']}")
    print(f"  global qubits:         {estimate['global_qubits']}")
    print(f"  amplitudes per rank:   {estimate['amplitudes_per_rank']:,}")
    print(f"  state memory per rank: {format_bytes(estimate['bytes_per_rank'])}")
    print()
    print("  Peak working set per rank, including the buffers the runtime")
    print("  allocates for pairwise exchanges:")
    print(f"    incoming shard buffer: {format_bytes(estimate['exchange_buffer_bytes'])}")
    print(f"    packing buffer:        {format_bytes(estimate['packing_buffer_bytes'])}")
    print(f"    peak per rank:         {format_bytes(estimate['peak_bytes_per_rank'])}")

    if estimate["system_memory_bytes"] is not None:
        print()
        print(
            f"  this host has {format_bytes(estimate['system_memory_bytes'])} of RAM; "
            f"all {estimate['ranks']} rank(s) on one node would "
            f"{'fit' if estimate['fits_on_this_host'] else 'NOT fit'}"
        )
    return 0


def _parse_rank_list(text: str) -> list[int]:
    ranks = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1 or value & (value - 1):
            raise SystemExit(f"rank counts must be powers of two, got {value}")
        ranks.append(value)
    if not ranks:
        raise SystemExit("no rank counts given")
    return ranks


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run a measurement sweep, or regenerate reports from existing raw data."""

    from aegisq.benchmark import report as report_module
    from aegisq.benchmark.runner import default_raw_path

    if args.benchmark_command == "pqc":
        from aegisq.benchmark.pqc import run_suite

        output = args.output or default_raw_path("pqc")
        print(f"Writing raw post-quantum measurements to {output}")
        run_suite(
            output,
            iterations=args.iterations,
            envelope_iterations=args.envelope_iterations,
            envelope_qubits=args.envelope_qubits,
        )
        print()
        print("Raw data written. Regenerate tables and plots with:")
        print("  aegisq benchmark report")
        return 0

    if args.benchmark_command == "search":
        from aegisq.benchmark.search import run_suite as run_search

        output = args.output or default_raw_path("search")
        print(f"Writing raw Grover measurements to {output}")
        run_search(
            output,
            bits=[int(b) for b in args.bits.split(",")],
            shots=args.shots,
            seed=args.seed,
        )
        print()
        print("Raw data written. Regenerate tables and plots with:")
        print("  aegisq benchmark report")
        return 0

    if args.benchmark_command == "report":
        written = report_module.write_reports(args.raw)
        if not written:
            print("no reports generated: no raw measurements found")
            return 1
        for name, path in written.items():
            print(f"  {name:<24} {path}")
        summary_path = report_module.PROCESSED_DIR / "summary.md"
        summary_path.write_text(report_module.markdown_summary(args.raw), encoding="utf-8")
        print(f"  {'markdown summary':<24} {summary_path}")
        return 0

    from aegisq.benchmark.communication import DEFAULT_FAMILIES, mapping_experiment
    from aegisq.benchmark.scaling import strong_scaling, weak_scaling

    output: Path = args.output or default_raw_path(
        {"strong": "strong_scaling", "weak": "weak_scaling", "mapping": "mapping_comparison"}[
            args.benchmark_command
        ]
    )
    ranks = _parse_rank_list(args.ranks)
    print(f"Writing raw measurements to {output}")

    if args.benchmark_command == "strong":
        strong_scaling(
            args.circuit,
            args.qubits,
            ranks,
            output,
            precision=args.precision,
            mapping=args.mapping,
            repeats=args.repeats,
            options=args.option,
            thread_policy=args.thread_policy,
        )
    elif args.benchmark_command == "weak":
        weak_scaling(
            args.circuit,
            args.qubits,
            ranks,
            output,
            precision=args.precision,
            mapping=args.mapping,
            repeats=args.repeats,
            options=args.option,
            thread_policy=args.thread_policy,
        )
    else:
        families = (
            [f.strip() for f in args.circuits.split(",")]
            if args.circuits
            else list(DEFAULT_FAMILIES)
        )
        per_family = {}
        for item in args.option or []:
            family, _, option = item.partition(":")
            per_family.setdefault(family, []).append(option)
        mapping_experiment(
            families,
            args.qubits,
            ranks,
            output,
            precision=args.precision,
            repeats=args.repeats,
            options=per_family,
            thread_policy=args.thread_policy,
            levers=tuple(lever.strip() for lever in args.levers.split(",")),
        )

    print()
    print("Raw data written. Regenerate tables and plots with:")
    print("  aegisq benchmark report")
    return 0


def _cmd_keys(args: argparse.Namespace) -> int:
    """Generate and inspect post-quantum identities."""
    from aegisq.secure.keys import (
        IdentityError,
        generate_client_identity,
        generate_cluster_identity,
        load_public_identity,
    )

    try:
        if args.keys_command == "init-client":
            identity = generate_client_identity(args.name, args.directory)
            print(f"Client identity '{args.name}' created")
            print(f"  signature algorithm: {identity.public.signature_algorithm}")
            print(f"  fingerprint:         {identity.public.signature_fingerprint}")
            print(f"  public key:          {identity.public_path}")
            print(f"  secret key:          {identity.secret_path}  (mode 0600)")
            print()
            print("  Share the public file with the cluster; never share or commit the secret.")
            return 0

        if args.keys_command == "init-cluster":
            identity = generate_cluster_identity(args.name, args.directory)
            print(f"Cluster identity '{args.name}' created")
            print(f"  KEM algorithm:       {identity.public.kem_algorithm}")
            print(f"  KEM fingerprint:     {identity.public.kem_fingerprint}")
            print(f"  signature algorithm: {identity.public.signature_algorithm}")
            print(f"  signature fingerprint: {identity.public.signature_fingerprint}")
            print(f"  public key:          {identity.public_path}")
            print(f"  secret key:          {identity.secret_path}  (mode 0600)")
            print()
            print("  Publish the public file so clients can encapsulate to this cluster.")
            return 0

        public = load_public_identity(args.path)
    except IdentityError as exc:
        raise SystemExit(str(exc)) from None

    print(f"{public.name} ({public.role}), created {public.created_at}")
    print(f"  signature: {public.signature_algorithm}  {public.signature_fingerprint}")
    if public.kem_public_key is not None:
        print(f"  KEM:       {public.kem_algorithm}  {public.kem_fingerprint}")
    return 0


def _cmd_secure_pack(args: argparse.Namespace) -> int:
    """Build a signed, encrypted job bundle."""
    from aegisq.secure.envelope import (
        EnvelopeError,
        ExecutionRequest,
        inspect_job,
        pack_job,
        write_job,
    )
    from aegisq.secure.keys import IdentityError, load_identity, load_public_identity

    circuit = _load_circuit(args.circuit, args.qubits, _parse_options(args.option))
    try:
        client = load_identity(args.identity)
        cluster_public = load_public_identity(args.cluster)
        raw = pack_job(
            circuit,
            ExecutionRequest(
                ranks=args.ranks,
                precision=args.precision,
                shots=args.shots,
                seed=args.seed,
                mapping_strategy=args.mapping,
            ),
            client,
            cluster_public,
        )
    except (IdentityError, EnvelopeError) as exc:
        raise SystemExit(str(exc)) from None

    output = args.output or Path(f"{circuit.name}.aqjob")
    write_job(output, raw)
    summary = inspect_job(raw)

    print(f"Packed {circuit.name} ({circuit.num_qubits} qubits, {len(circuit)} gates)")
    print(f"  job id:            {summary['job_id']}")
    print(f"  signed by:         {summary['client_name']} ({summary['client_fingerprint']})")
    print(f"  addressed to:      {summary['cluster_name']} ({summary['cluster_fingerprint']})")
    print(
        f"  suite:             {summary['crypto_suite']['kem']} + "
        f"{summary['crypto_suite']['signature']} + {summary['crypto_suite']['aead']}"
    )
    print(f"  ciphertext:        {summary['ciphertext_bytes']} bytes")
    print(f"  bundle:            {output} ({len(raw)} bytes)")
    return 0


def _cmd_secure_inspect(args: argparse.Namespace) -> int:
    """Show a bundle's public metadata without decrypting it."""
    import json

    from aegisq.secure.envelope import EnvelopeError, inspect_job, read_job

    try:
        summary = inspect_job(read_job(args.bundle))
    except EnvelopeError as exc:
        raise SystemExit(str(exc)) from None

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"Job bundle: {args.bundle}")
    print(f"  job id:            {summary['job_id']}")
    print(f"  created:           {summary['created_at']}")
    print(f"  signed by:         {summary['client_name']} ({summary['client_fingerprint']})")
    print(f"  addressed to:      {summary['cluster_name']} ({summary['cluster_fingerprint']})")
    suite = summary["crypto_suite"]
    print(f"  KEM:               {suite['kem']}")
    print(f"  signature:         {suite['signature']}")
    print(f"  KDF / AEAD:        {suite['kdf']} / {suite['aead']}")
    print(f"  ciphertext:        {summary['ciphertext_bytes']} bytes")
    print(f"  envelope sha256:   {summary['envelope_sha256']}")
    print()
    print("  Metadata only: the signature has not been checked and the payload")
    print("  stays encrypted. Use `aegisq secure-run` to verify and execute.")
    return 0


def _cmd_secure_run(args: argparse.Namespace) -> int:
    """Verify, decrypt and execute an authenticated job bundle.

    The order of checks is the point: nothing inside the envelope is trusted
    until the signature verifies against a key the cluster already knows.
    """
    import json

    from aegisq.runtime import Simulator
    from aegisq.runtime.distributed import is_distributed, preferred_backend, rank, world_size
    from aegisq.secure.envelope import EnvelopeError, open_job, read_job
    from aegisq.secure.keys import IdentityError, load_identity, load_trusted_directory
    from aegisq.secure.replay import ReplayDatabase, ReplayDetected
    from aegisq.secure.signatures import SignatureError

    is_lead = rank() == 0

    try:
        cluster = load_identity(args.cluster)
        trusted = load_trusted_directory(args.trusted)
        opened = open_job(read_job(args.bundle), cluster, trusted)
    except (IdentityError, EnvelopeError, SignatureError) as exc:
        raise SystemExit(f"job rejected: {exc}") from None

    manifest = opened.manifest
    database = ReplayDatabase(args.replay_db)
    try:
        # Every rank refuses a replayed job; only rank 0 records it, so the
        # ranks cannot disagree and deadlock.
        database.check(manifest.job_id)
        if is_lead:
            database.record(
                manifest.job_id,
                manifest.nonce,
                opened.client.signature_fingerprint,
                manifest.created_at,
            )
    except ReplayDetected as exc:
        raise SystemExit(f"job rejected: {exc}") from None

    requested_ranks = manifest.execution.ranks
    actual_ranks = world_size()
    if is_lead and requested_ranks != actual_ranks:
        print(
            f"note: job requested {requested_ranks} rank(s) but this world has "
            f"{actual_ranks}; running as launched and recording both",
            file=sys.stderr,
        )

    options: dict[str, object] = {}
    placement = None
    if manifest.execution.mapping_strategy == "optimized" and is_distributed():
        from aegisq.compiler import optimize_placement

        placement = optimize_placement(opened.circuit, actual_ranks, manifest.execution.precision)
        options["mapping"] = list(placement.mapping)

    backend = preferred_backend()
    simulator = Simulator(backend, precision=manifest.execution.precision, **options)
    result = simulator.run(
        opened.circuit,
        shots=manifest.execution.shots,
        seed=manifest.execution.seed,
        save_statevector=False,
    )

    if not is_lead:
        return 0

    payload = {
        "job_id": manifest.job_id,
        "client": {
            "name": opened.client.name,
            "fingerprint": opened.client.signature_fingerprint,
        },
        "circuit": {
            "name": manifest.circuit_name,
            "sha256": manifest.circuit_sha256,
            "num_qubits": manifest.num_qubits,
            "gates": manifest.gate_count,
            "depth": manifest.depth,
        },
        "execution": {
            "requested_ranks": requested_ranks,
            "world_size": actual_ranks,
            "backend": backend,
            "precision": manifest.execution.precision,
            "shots": manifest.execution.shots,
            "seed": manifest.execution.seed,
            "mapping_strategy": manifest.execution.mapping_strategy,
            "global_qubits": list(placement.global_qubits) if placement else None,
        },
        "counts": result.counts,
        "metrics": {k: v for k, v in result.metrics.items() if k != "per_opcode"},
    }

    # Sign the execution record with the cluster identity, so the result can
    # be attributed and tamper-checked later.
    from aegisq.provenance import (
        artifacts_from_result,
        build_result_manifest,
        sign_result,
        write_result,
    )

    metrics = {k: v for k, v in result.metrics.items() if k != "per_opcode"}
    result_manifest = build_result_manifest(
        job_id=manifest.job_id,
        input_circuit_sha256=manifest.circuit_sha256,
        artifacts=artifacts_from_result(result.counts, metrics),
        execution=payload["execution"],
        performance={
            "wall_seconds": metrics.get("wall_seconds", 0.0),
            "compute_seconds": metrics.get("compute_seconds", 0.0),
        },
        communication={
            "communication_seconds": metrics.get("communication_seconds", 0.0),
            "bytes_sent": metrics.get("bytes_sent", 0),
            "bytes_received": metrics.get("bytes_received", 0),
            "pairwise_exchanges": metrics.get("pairwise_exchanges", 0),
        },
        client_fingerprint=opened.client.signature_fingerprint,
    )
    signed = sign_result(result_manifest, cluster)
    payload["result_manifest"] = {
        "output_merkle_root": result_manifest.output_merkle_root,
        "signed_by": cluster.public.signature_fingerprint,
    }

    if args.output:
        write_result(args.output, signed)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Job {manifest.job_id} accepted")
    print(f"  signed by:      {opened.client.name} ({opened.client.signature_fingerprint})")
    print(
        f"  circuit:        {manifest.circuit_name} "
        f"({manifest.num_qubits} qubits, {manifest.gate_count} gates)"
    )
    print(f"  circuit sha256: {manifest.circuit_sha256}")
    print(
        f"  executed on:    {actual_ranks} rank(s), backend {backend}, "
        f"{manifest.execution.precision}"
    )
    if placement is not None:
        print(f"  placement:      global qubits {list(placement.global_qubits)}")
    print()
    if result.counts:
        total = sum(result.counts.values())
        print(f"Counts ({total} shots, top {args.top}):")
        for bitstring, count in result.most_frequent(args.top):
            print(f"  {bitstring}  {count:>8}  {count / total * 100:6.2f}%")
        print()
    print(f"  output merkle root: {result_manifest.output_merkle_root}")
    print(f"  signed by cluster:  {cluster.public.signature_fingerprint}")
    if args.output:
        print(f"  signed result:      {args.output}")
    else:
        print("  (pass --output to keep the signed result bundle)")
    return 0


def _cmd_verify_result(args: argparse.Namespace) -> int:
    """Check a signed result bundle."""
    import json

    from aegisq.provenance import ProvenanceError, read_result, verify_result
    from aegisq.secure.keys import IdentityError, load_public_identity

    try:
        cluster_public = load_public_identity(args.cluster)
        raw = read_result(args.result)
    except (IdentityError, ProvenanceError) as exc:
        raise SystemExit(str(exc)) from None

    report = verify_result(
        raw,
        cluster_public,
        artifact_directory=args.artifacts,
        expected_job_id=args.job_id,
        expected_circuit_sha256=args.circuit_sha256,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "job_id": report.job_id,
                    "cluster_name": report.cluster_name,
                    "cluster_fingerprint": report.cluster_fingerprint,
                    "checks": [
                        {"name": name, "ok": ok, "detail": detail}
                        for name, ok, detail in report.checks
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(report.summary())
    return 0 if report.ok else 1


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

    run = subparsers.add_parser(
        "run",
        help="execute a circuit (distributed when launched under mpirun)",
        description=(
            "Execute a circuit. Under `mpirun -np P` the state vector is "
            "partitioned across the P ranks automatically."
        ),
    )
    run.add_argument(
        "circuit",
        help="circuit file (.qasm, .json) or a benchmark family name",
    )
    run.add_argument("--qubits", type=int, help="width, when building a named family")
    run.add_argument("--shots", type=int, default=1024, help="measurement shots (0 to skip)")
    run.add_argument("--seed", type=int, default=42, help="sampling seed")
    run.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    run.add_argument(
        "--backend",
        choices=("auto", "cpp", "mpi", "reference"),
        default="auto",
        help="auto selects mpi when the world has more than one rank",
    )
    run.add_argument(
        "--optimize",
        action="store_true",
        help="apply a communication-aware qubit placement before running",
    )
    run.add_argument(
        "--statevector",
        action="store_true",
        help="print non-zero amplitudes (gathers the full state; small circuits only)",
    )
    run.add_argument("--top", type=int, default=10, help="how many outcomes to print")
    run.add_argument("--option", action="append", metavar="KEY=VALUE", help="extra family option")
    run.add_argument("--json", action="store_true", help="emit machine-readable output")
    run.set_defaults(func=_cmd_run)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="run measurement sweeps and regenerate reports",
        description=(
            "Every figure in this repository is produced from raw CSV written "
            "by these sweeps. Sweeps launch one mpirun per configuration."
        ),
    )
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)

    strong = benchmark_sub.add_parser("strong", help="fixed problem size, growing rank count")
    strong.add_argument("--circuit", required=True)
    strong.add_argument("--qubits", type=int, required=True)
    strong.add_argument("--ranks", default="1,2,4,8")
    strong.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    strong.add_argument("--mapping", choices=("default", "optimized"), default="default")
    strong.add_argument("--repeats", type=int, default=3)
    strong.add_argument("--option", action="append", metavar="KEY=VALUE")
    strong.add_argument(
        "--thread-policy",
        choices=("one-thread-per-rank", "fixed-total-cores"),
        default="one-thread-per-rank",
        help="one-thread-per-rank grows total cores with ranks; "
        "fixed-total-cores keeps ranks x threads at the core count",
    )
    strong.add_argument("--output", type=Path)
    strong.set_defaults(func=_cmd_benchmark)

    weak = benchmark_sub.add_parser(
        "weak", help="problem grows with the rank count (constant shard size)"
    )
    weak.add_argument("--circuit", required=True)
    weak.add_argument("--qubits", type=int, required=True, help="width at one rank")
    weak.add_argument("--ranks", default="1,2,4,8")
    weak.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    weak.add_argument("--mapping", choices=("default", "optimized"), default="default")
    weak.add_argument("--repeats", type=int, default=3)
    weak.add_argument("--option", action="append", metavar="KEY=VALUE")
    weak.add_argument(
        "--thread-policy",
        choices=("one-thread-per-rank", "fixed-total-cores"),
        default="one-thread-per-rank",
        help="one-thread-per-rank grows total cores with ranks; "
        "fixed-total-cores keeps ranks x threads at the core count",
    )
    weak.add_argument("--output", type=Path)
    weak.set_defaults(func=_cmd_benchmark)

    mapping_cmd = benchmark_sub.add_parser(
        "mapping", help="default versus communication-aware placement"
    )
    mapping_cmd.add_argument("--circuits", help="comma-separated families")
    mapping_cmd.add_argument("--qubits", type=int, required=True)
    mapping_cmd.add_argument("--ranks", default="2,4,8")
    mapping_cmd.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    mapping_cmd.add_argument("--repeats", type=int, default=3)
    mapping_cmd.add_argument(
        "--option", action="append", metavar="FAMILY:KEY=VALUE", help="per-family option"
    )
    mapping_cmd.add_argument(
        "--levers",
        default="placement",
        help="comma-separated optimisation levers to sweep: placement, fusion",
    )
    mapping_cmd.add_argument(
        "--thread-policy",
        choices=("one-thread-per-rank", "fixed-total-cores"),
        default="fixed-total-cores",
        help="one-thread-per-rank grows total cores with ranks; "
        "fixed-total-cores keeps ranks x threads at the core count",
    )
    mapping_cmd.add_argument("--output", type=Path)
    mapping_cmd.set_defaults(func=_cmd_benchmark)

    pqc_cmd = benchmark_sub.add_parser(
        "pqc",
        help="measure ML-KEM / ML-DSA primitives and the end-to-end envelope",
    )
    pqc_cmd.add_argument("--iterations", type=int, default=1000)
    pqc_cmd.add_argument("--envelope-iterations", type=int, default=50)
    pqc_cmd.add_argument("--envelope-qubits", type=int, default=20)
    pqc_cmd.add_argument("--output", type=Path)
    pqc_cmd.set_defaults(func=_cmd_benchmark)

    search_cmd = benchmark_sub.add_parser(
        "search",
        help="measure Grover's oracle-query scaling against classical search",
    )
    search_cmd.add_argument("--bits", default="2,3,4,5,6,7,8", help="search-space sizes as 2^k")
    search_cmd.add_argument("--shots", type=int, default=2048)
    search_cmd.add_argument("--seed", type=int, default=42)
    search_cmd.add_argument("--output", type=Path)
    search_cmd.set_defaults(func=_cmd_benchmark)

    report_cmd = benchmark_sub.add_parser(
        "report", help="regenerate processed tables and plots from raw measurements"
    )
    report_cmd.add_argument("--raw", type=Path, help="raw CSV file or directory")
    report_cmd.set_defaults(func=_cmd_benchmark)

    keys = subparsers.add_parser(
        "keys",
        help="generate and inspect post-quantum identities",
        description=(
            "ML-KEM-768 (FIPS 203) and ML-DSA-65 (FIPS 204) key material, from "
            "liboqs. Secret files are written with owner-only permissions and "
            "are excluded by .gitignore."
        ),
    )
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)

    init_client = keys_sub.add_parser("init-client", help="create a job-signing identity")
    init_client.add_argument("name")
    init_client.add_argument("--directory", type=Path, default=Path("keys"))
    init_client.set_defaults(func=_cmd_keys)

    init_cluster = keys_sub.add_parser(
        "init-cluster", help="create a cluster identity (KEM + signing)"
    )
    init_cluster.add_argument("name")
    init_cluster.add_argument("--directory", type=Path, default=Path("keys"))
    init_cluster.set_defaults(func=_cmd_keys)

    fingerprint_cmd = keys_sub.add_parser(
        "fingerprint", help="show the fingerprints of a public identity file"
    )
    fingerprint_cmd.add_argument("path", type=Path)
    fingerprint_cmd.set_defaults(func=_cmd_keys)

    secure_pack = subparsers.add_parser(
        "secure-pack",
        help="build a signed, encrypted job bundle",
        description=(
            "Encapsulate to the cluster's ML-KEM key, derive an AES-256-GCM key "
            "bound to this job, encrypt the manifest and circuit together, and "
            "sign the result with the client's ML-DSA key."
        ),
    )
    secure_pack.add_argument("circuit", help="circuit file or benchmark family name")
    secure_pack.add_argument("--qubits", type=int, help="width, when building a named family")
    secure_pack.add_argument(
        "--identity", type=Path, required=True, help="client identity base path (no suffix)"
    )
    secure_pack.add_argument(
        "--cluster", type=Path, required=True, help="cluster public identity file"
    )
    secure_pack.add_argument("--ranks", type=int, default=1)
    secure_pack.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    secure_pack.add_argument("--shots", type=int, default=1024)
    secure_pack.add_argument("--seed", type=int, default=42)
    secure_pack.add_argument("--mapping", choices=("default", "optimized"), default="default")
    secure_pack.add_argument("--option", action="append", metavar="KEY=VALUE")
    secure_pack.add_argument("--output", type=Path)
    secure_pack.set_defaults(func=_cmd_secure_pack)

    secure_inspect = subparsers.add_parser(
        "secure-inspect",
        help="show a bundle's public metadata without decrypting it",
    )
    secure_inspect.add_argument("bundle", type=Path)
    secure_inspect.add_argument("--json", action="store_true")
    secure_inspect.set_defaults(func=_cmd_secure_inspect)

    secure_run = subparsers.add_parser(
        "secure-run",
        help="verify, decrypt and execute a job bundle",
        description=(
            "Checks in order: envelope structure, client key against the "
            "cluster's trusted set, ML-DSA signature, addressee, replay state, "
            "then decryption and the circuit hash. Nothing inside the envelope "
            "is trusted until the signature verifies."
        ),
    )
    secure_run.add_argument("bundle", type=Path)
    secure_run.add_argument(
        "--cluster", type=Path, required=True, help="cluster identity base path"
    )
    secure_run.add_argument(
        "--trusted",
        type=Path,
        required=True,
        help="directory of trusted client public identities",
    )
    secure_run.add_argument(
        "--replay-db", type=Path, default=Path("replay_db.json"), help="replay state file"
    )
    secure_run.add_argument("--output", type=Path, help="write the signed .aqresult bundle here")
    secure_run.add_argument("--top", type=int, default=10)
    secure_run.add_argument("--json", action="store_true")
    secure_run.set_defaults(func=_cmd_secure_run)

    verify_result_cmd = subparsers.add_parser(
        "verify-result",
        help="check a signed result bundle",
        description=(
            "Verifies the cluster signature, the Merkle root over the artefact "
            "list and each artefact's hashes. A valid signature authenticates "
            "the origin of a record and detects tampering; it is not evidence "
            "that the computation was performed correctly."
        ),
    )
    verify_result_cmd.add_argument("result", type=Path)
    verify_result_cmd.add_argument(
        "--cluster", type=Path, required=True, help="cluster public identity file"
    )
    verify_result_cmd.add_argument(
        "--artifacts", type=Path, help="directory holding referenced (non-inline) artefacts"
    )
    verify_result_cmd.add_argument("--job-id", help="expected job id, to check linkage")
    verify_result_cmd.add_argument("--circuit-sha256", help="expected input circuit hash")
    verify_result_cmd.add_argument("--json", action="store_true")
    verify_result_cmd.set_defaults(func=_cmd_verify_result)

    estimate = subparsers.add_parser(
        "estimate",
        help="report the memory a distributed state vector would need",
    )
    estimate.add_argument("--qubits", type=int, required=True)
    estimate.add_argument("--ranks", type=int, default=1, help="MPI world size (power of two)")
    estimate.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    estimate.add_argument("--json", action="store_true")
    estimate.set_defaults(func=_cmd_estimate)

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
