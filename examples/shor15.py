#!/usr/bin/env python3
"""Shor's algorithm factoring 15 (and 21) by simulation.

This is an educational problem size. Factoring a cryptographically relevant
RSA modulus needs thousands of logical qubits and error correction; simulating
even a few dozen qubits costs exponential memory, which is the premise of this
whole project.

$ python examples/shor15.py
$ python examples/shor15.py --modulus 21 --counting 6
"""

from __future__ import annotations

import argparse

from aegisq.algorithms.shor import factor, run_shor, shor_circuit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulus", type=int, default=15)
    parser.add_argument("--base", type=int, help="fix the base instead of searching")
    parser.add_argument("--counting", type=int, default=None, help="counting qubits")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.base is not None:
        built = shor_circuit(args.modulus, args.base, args.counting)
        print(
            f"circuit: {built.num_qubits} qubits "
            f"({len(built.counting_qubits)} counting, {len(built.work_qubits)} work, "
            f"{len(built.ancilla_qubits)} ancilla), {len(built.circuit)} gates"
        )
        print()
        result = run_shor(
            args.modulus,
            args.base,
            shots=args.shots,
            seed=args.seed,
            counting_qubits=args.counting,
        )
    else:
        result = factor(
            args.modulus, shots=args.shots, seed=args.seed, counting_qubits=args.counting
        )

    print(result.summary())
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
