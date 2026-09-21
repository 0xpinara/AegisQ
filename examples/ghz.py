#!/usr/bin/env python3
"""GHZ state preparation, run locally or across MPI ranks.

$ python examples/ghz.py
$ mpirun -np 4 python examples/ghz.py --qubits 24
"""

from __future__ import annotations

import argparse

from aegisq.algorithms import ghz
from aegisq.runtime import Simulator
from aegisq.runtime.distributed import preferred_backend, rank, world_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=12)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    circuit = ghz(args.qubits).measure_all()
    result = Simulator(preferred_backend()).run(circuit, shots=args.shots, seed=args.seed)

    if rank() != 0:
        return 0

    print(f"GHZ state over {args.qubits} qubits on {world_size()} rank(s)")
    for bitstring, count in result.most_frequent(4):
        print(f"  {bitstring}  {count:>6}  {count / args.shots * 100:5.1f}%")

    # Only two outcomes should ever appear.
    assert set(result.counts) <= {"0" * args.qubits, "1" * args.qubits}

    if "bytes_sent" in result.metrics:
        print()
        print(f"  MPI bytes sent:     {result.metrics['bytes_sent']:,}")
        print(f"  pairwise exchanges: {result.metrics['pairwise_exchanges']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
