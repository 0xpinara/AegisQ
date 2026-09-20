#!/usr/bin/env python3
"""Grover search over a small unstructured space.

Demonstrates the quadratic query advantage that motivates doubling symmetric
key sizes in post-quantum guidance. It says nothing about breaking AES.

    python examples/grover.py --bits 6 --marked 37
"""

from __future__ import annotations

import argparse

from aegisq.algorithms.grover import grover, optimal_iterations
from aegisq.benchmark.search import theoretical_success
from aegisq.runtime import Simulator
from aegisq.runtime.distributed import preferred_backend, rank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, default=5, help="search space is 2**bits")
    parser.add_argument("--marked", type=int, default=19)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    space = 2**args.bits
    marked = args.marked % space
    iterations = optimal_iterations(args.bits)
    circuit = grover(args.bits, marked=marked, iterations=iterations)

    result = Simulator(preferred_backend()).run(circuit, shots=args.shots, seed=args.seed)
    if rank() != 0:
        return 0

    key = format(marked, f"0{args.bits}b")
    hits = result.counts.get(key, 0)

    print(f"Grover search over {space} items, marked state {marked} ({key})")
    print(f"  circuit:            {circuit.num_qubits} qubits, {len(circuit)} gates")
    print(f"  oracle queries:     {iterations}")
    print(f"  classical expected: {(space + 1) / 2:.1f}")
    print()
    print(f"  measured success:   {hits / args.shots * 100:.1f}%")
    print(f"  theoretical:        {theoretical_success(args.bits, iterations) * 100:.1f}%")
    print()
    print("  top outcomes:")
    for bitstring, count in result.most_frequent(3):
        marker = "  <- marked" if bitstring == key else ""
        print(f"    {bitstring}  {count:>6}  {count / args.shots * 100:5.1f}%{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
