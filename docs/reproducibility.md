# Reproducibility

## What is deterministic

| Quantity | Reproducible when | Notes |
|---|---|---|
| final state vector | always | up to floating-point associativity; fp64 agreement with the reference backend is asserted at 1e-11 |
| measurement counts | same circuit, shots, seed, backend and qubit placement | the sampler draws from a seeded MT19937-64 and walks accumulated probability mass |
| counts across rank counts | same placement | the draw sequence depends only on `(shots, seed)`, so 1 rank and 8 ranks produce identical counts |
| communication volume | always | a deterministic function of the circuit and the placement — the MPI suite asserts the exact byte counts |
| wall time | never exactly | reported as best-of-repeats with the spread, on a named host |

## What is *not* bit-identical, and why

- **Counts across different qubit placements.** A permuted placement changes
  the order in which basis states are visited while the probability mass is
  accumulated, so a given draw lands on a different index. The distribution is
  unchanged; the specific assignment is not. The placement is recorded in the
  result manifest for this reason.
- **Counts across backends.** The NumPy reference backend samples with
  `Generator.multinomial`; the native backend uses the partition-independent
  scan described above. Each is reproducible on its own terms, and the backend
  is recorded with every result.
- **fp32 runs.** Single-precision accumulates visibly more rounding error;
  cross-backend comparisons use a correspondingly looser tolerance (1e-5).

## Environment capture

Every benchmark row and every signed result manifest records the host, CPU,
core count, OS, compiler, MPI library, git commit and whether the working tree
was dirty at the time. A measurement taken from a dirty tree is marked as such
rather than quietly attributed to the last commit.

## Rebuilding everything from raw data

```bash
aegisq benchmark report        # regenerates processed tables and plots
```

Deleting `benchmarks/processed/` and `benchmarks/plots/` and re-running that
command must reproduce them exactly from `benchmarks/raw/`. If it does not,
something has been hand-edited, which is a bug.
