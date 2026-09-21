#!/usr/bin/env python3
"""Quantum Fourier transform, with and without communication-aware placement.

The QFT is the circuit where qubit placement matters most: every controlled
phase targets the higher-indexed qubit, and the default placement puts exactly
those qubits on the rank-selecting positions.

$ mpirun -np 8 python examples/qft.py --qubits 24
"""

from __future__ import annotations

import argparse

from aegisq.algorithms import qft
from aegisq.compiler import optimize_placement
from aegisq.runtime import Simulator
from aegisq.runtime.distributed import is_distributed, preferred_backend, rank, world_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--shots", type=int, default=0)
    args = parser.parse_args()

    circuit = qft(args.qubits)
    backend = preferred_backend()
    is_lead = rank() == 0

    if is_lead:
        print(f"QFT on {args.qubits} qubits: {len(circuit)} gates, depth {circuit.depth()}")
        print(f"backend {backend}, {world_size()} rank(s)")
        print()

    baseline = Simulator(backend).run(circuit, shots=args.shots, save_statevector=False)

    if not is_distributed():
        if is_lead:
            print(f"wall time: {baseline.metrics['wall_seconds'] * 1000:.1f} ms")
            print("(run under mpirun to see the placement comparison)")
        return 0

    placement = optimize_placement(circuit, world_size())
    optimized = Simulator(backend, mapping=list(placement.mapping)).run(
        circuit, shots=args.shots, save_statevector=False
    )

    if not is_lead:
        return 0

    for label, result in (("default placement", baseline), ("communication-aware", optimized)):
        print(f"{label}:")
        print(f"  wall time:      {result.metrics['wall_seconds'] * 1000:8.1f} ms")
        print(f"  MPI bytes sent: {result.metrics['bytes_sent']:>12,}")
        print(f"  exchanges:      {result.metrics['pairwise_exchanges']:>12,}")

    saved = baseline.metrics["bytes_sent"] - optimized.metrics["bytes_sent"]
    if baseline.metrics["bytes_sent"]:
        print()
        print(f"measured reduction: {saved / baseline.metrics['bytes_sent'] * 100:.1f}% of bytes")
        print(f"chosen global qubits: {list(placement.global_qubits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
