#!/usr/bin/env python3
"""Trotterised transverse-field Ising evolution.

    python examples/ising.py --qubits 14 --steps 6
    mpirun -np 4 python examples/ising.py --qubits 22 --steps 4
"""

from __future__ import annotations

import argparse

from aegisq.algorithms import ising_trotter
from aegisq.runtime import Simulator
from aegisq.runtime.distributed import preferred_backend, rank, world_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=12)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    circuit = ising_trotter(args.qubits, steps=args.steps).measure_all()
    result = Simulator(preferred_backend()).run(circuit, shots=args.shots, seed=args.seed)

    if rank() != 0:
        return 0

    print(
        f"Ising chain: {args.qubits} sites, {args.steps} Trotter steps, "
        f"{len(circuit)} gates on {world_size()} rank(s)"
    )
    print(f"wall time: {result.metrics['wall_seconds'] * 1000:.1f} ms")
    print()
    print("most frequent configurations:")
    for bitstring, count in result.most_frequent(8):
        print(f"  {bitstring}  {count:>6}  {count / args.shots * 100:5.2f}%")
    print()
    print(f"distinct outcomes observed: {len(result.counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
