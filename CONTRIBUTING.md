# Contributing to AegisQ-HPC

## Development loop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,crypto,benchmark,validation]"
make build
make test
make test-mpi      # requires mpirun
```

## Ground rules

1. **Never commit a benchmark number that was not measured.** Everything under
   `benchmarks/processed/` and `benchmarks/plots/` must be derivable from a raw
   CSV in `benchmarks/raw/` produced by an actual run.
2. **Never invent cryptography.** ML-KEM and ML-DSA come from liboqs; AES-GCM
   and HKDF come from the `cryptography` package.
3. **Correctness before performance.** A distributed kernel lands only with a
   test that cross-checks it against the single-process reference simulator.
4. **Optional dependencies stay optional.** Missing MPI, liboqs, Qiskit or CUDA
   must degrade to a clear message, never a crash.
5. **Document the threat model honestly.** If a change alters what the security
   layer does or does not protect, update `docs/security-model.md` in the same
   commit.

## Style

- Python: `ruff check` and `ruff format` (line length 100).
- C++: `clang-format` with the repository `.clang-format` (Google base, 4-space
  indent, 100 columns).
- Commit messages: `type: imperative summary`, e.g. `feat: add distributed CNOT`.

## Tests

| Directory | Contents |
|---|---|
| `tests/unit` | pure-Python and single-process behaviour |
| `tests/integration` | end-to-end CLI and job flows |
| `tests/mpi` | distributed correctness, launched via `scripts/run_mpi_tests.sh` |
| `tests/crypto` | envelope, signature, replay and tamper tests |
| `tests/cross_validation` | randomised comparisons against Qiskit |

Mark long tests with `@pytest.mark.slow` so the default developer loop stays fast.
