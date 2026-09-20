## Measurement host

- host: `pias-MacBook-Pro.local`
- CPU: Apple M2 (8 logical cores)
- OS: macOS-15.6.1-arm64-arm-64bit
- compiler: AppleClang 17.0.0.17000603
- MPI: Open MPI v5.0.10, package: Open MPI brew@Sequoia-arm64.local Distribution, ident: 5.0.10, repo rev: v5.0.10, Feb 23, 2026
- AegisQ 0.1.0 at commit `4bf173533b4a`

## Communication-aware placement (measured)

| circuit | qubits | ranks | baseline MPI bytes | optimized MPI bytes | reduction | baseline wall (s) | optimized wall (s) | wall change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ghz | 20 | 2 | 8,388,608 | 8,388,608 | 0.0% | 0.008 | 0.008 | +2.7% |
| ghz | 20 | 4 | 16,777,216 | 16,777,216 | 0.0% | 0.009 | 0.009 | -1.3% |
| ghz | 20 | 8 | 25,165,824 | 25,165,824 | 0.0% | 0.013 | 0.014 | +8.7% |
| grover | 20 | 2 | 536,870,912 | 150,994,944 | 71.9% | 0.460 | 0.436 | -5.1% |
| grover | 20 | 4 | 1,207,959,552 | 301,989,888 | 75.0% | 0.474 | 0.416 | -12.4% |
| grover | 20 | 8 | 1,879,048,192 | 520,093,696 | 72.3% | 0.529 | 0.439 | -16.9% |
| ising | 20 | 2 | 150,994,944 | 83,886,080 | 44.4% | 0.128 | 0.121 | -5.7% |
| ising | 20 | 4 | 301,989,888 | 234,881,024 | 22.2% | 0.129 | 0.123 | -5.0% |
| ising | 20 | 8 | 452,984,832 | 385,875,968 | 14.8% | 0.162 | 0.150 | -6.9% |
| qft | 20 | 2 | 25,165,824 | 25,165,824 | 0.0% | 0.306 | 0.315 | +2.7% |
| qft | 20 | 4 | 67,108,864 | 67,108,864 | 0.0% | 0.299 | 0.299 | +0.0% |
| qft | 20 | 8 | 125,829,120 | 125,829,120 | 0.0% | 0.256 | 0.253 | -1.1% |
| random | 20 | 2 | 159,383,552 | 83,886,080 | 47.4% | 0.103 | 0.099 | -4.6% |
| random | 20 | 4 | 251,658,240 | 167,772,160 | 33.3% | 0.110 | 0.098 | -11.0% |
| random | 20 | 8 | 411,041,792 | 251,658,240 | 38.8% | 0.139 | 0.131 | -5.9% |

## Strong scaling (measured)

| circuit | qubits | thread policy | ranks | threads/rank | wall (s) | speedup | efficiency |
|---|---:|---|---:|---:|---:|---:|---:|
| ising | 22 | one-thread-per-rank | 1 | 1 | 1.895 | 1.00x | 100% |
| ising | 22 | one-thread-per-rank | 2 | 1 | 1.006 | 1.88x | 94% |
| ising | 22 | one-thread-per-rank | 4 | 1 | 0.715 | 2.65x | 66% |
| ising | 22 | one-thread-per-rank | 8 | 1 | 0.725 | 2.61x | 33% |
| qft | 22 | fixed-total-cores | 1 | 8 | 2.052 | 1.00x | 100% |
| qft | 22 | fixed-total-cores | 2 | 4 | 1.893 | 1.08x | 54% |
| qft | 22 | fixed-total-cores | 4 | 2 | 1.902 | 1.08x | 27% |
| qft | 22 | fixed-total-cores | 8 | 1 | 1.799 | 1.14x | 14% |
| qft | 22 | one-thread-per-rank | 1 | 1 | 5.313 | 1.00x | 100% |
| qft | 22 | one-thread-per-rank | 2 | 1 | 2.976 | 1.79x | 89% |
| qft | 22 | one-thread-per-rank | 4 | 1 | 2.017 | 2.63x | 66% |
| qft | 22 | one-thread-per-rank | 8 | 1 | 1.829 | 2.90x | 36% |

## Weak scaling (measured)

| circuit | thread policy | ranks | qubits | amplitudes/rank | wall (s) | efficiency |
|---|---|---:|---:|---:|---:|---:|
| ising | one-thread-per-rank | 1 | 20 | 1,048,576 | 0.419 | 100% |
| ising | one-thread-per-rank | 2 | 21 | 1,048,576 | 0.478 | 88% |
| ising | one-thread-per-rank | 4 | 22 | 1,048,576 | 0.733 | 57% |
| ising | one-thread-per-rank | 8 | 23 | 1,048,576 | 1.531 | 27% |

## Cost-model accuracy

- 42 of 42 distributed configurations sent exactly the number of bytes the cost model predicted.
