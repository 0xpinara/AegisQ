## Measurement host

- host: `pias-MacBook-Pro.local`
- CPU: Apple M2 (8 logical cores)
- OS: macOS-15.6.1-arm64-arm-64bit
- compiler: AppleClang 17.0.0.17000603
- MPI: Open MPI v5.0.10, package: Open MPI brew@Sequoia-arm64.local Distribution, ident: 5.0.10, repo rev: v5.0.10, Feb 23, 2026
- AegisQ 0.1.0 at commit `74a1a238513b`

## Communication-aware placement (measured)

| circuit | qubits | ranks | baseline MPI bytes | optimized MPI bytes | reduction | baseline wall (s) | optimized wall (s) | wall change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ghz | 20 | 2 | 8,388,608 | 8,388,608 | 0.0% | 0.008 | 0.008 | +5.8% |
| ghz | 20 | 4 | 16,777,216 | 16,777,216 | 0.0% | 0.008 | 0.009 | +16.8% |
| ghz | 20 | 8 | 25,165,824 | 25,165,824 | 0.0% | 0.014 | 0.013 | -7.9% |
| grover | 20 | 2 | 536,870,912 | 150,994,944 | 71.9% | 0.469 | 0.428 | -8.8% |
| grover | 20 | 4 | 1,207,959,552 | 301,989,888 | 75.0% | 0.526 | 0.479 | -8.9% |
| grover | 20 | 8 | 1,879,048,192 | 520,093,696 | 72.3% | 0.582 | 0.461 | -20.7% |
| ising | 20 | 2 | 150,994,944 | 83,886,080 | 44.4% | 0.148 | 0.123 | -17.2% |
| ising | 20 | 4 | 301,989,888 | 234,881,024 | 22.2% | 0.130 | 0.123 | -5.1% |
| ising | 20 | 8 | 452,984,832 | 385,875,968 | 14.8% | 0.160 | 0.161 | +1.1% |
| qft | 20 | 2 | 343,932,928 | 25,165,824 | 92.7% | 0.345 | 0.313 | -9.2% |
| qft | 20 | 4 | 671,088,640 | 67,108,864 | 90.0% | 0.338 | 0.296 | -12.5% |
| qft | 20 | 8 | 981,467,136 | 125,829,120 | 87.2% | 0.349 | 0.264 | -24.3% |
| random | 20 | 2 | 159,383,552 | 83,886,080 | 47.4% | 0.105 | 0.099 | -6.2% |
| random | 20 | 4 | 251,658,240 | 167,772,160 | 33.3% | 0.106 | 0.097 | -8.3% |
| random | 20 | 8 | 411,041,792 | 251,658,240 | 38.8% | 0.155 | 0.124 | -20.0% |

## Strong scaling (measured)

| circuit | qubits | thread policy | ranks | threads/rank | wall (s) | speedup | efficiency |
|---|---:|---|---:|---:|---:|---:|---:|
| ising | 22 | one-thread-per-rank | 1 | 1 | 1.896 | 1.00x | 100% |
| ising | 22 | one-thread-per-rank | 2 | 1 | 1.031 | 1.84x | 92% |
| ising | 22 | one-thread-per-rank | 4 | 1 | 0.742 | 2.56x | 64% |
| ising | 22 | one-thread-per-rank | 8 | 1 | 0.793 | 2.39x | 30% |
| qft | 22 | fixed-total-cores | 1 | 8 | 2.119 | 1.00x | 100% |
| qft | 22 | fixed-total-cores | 2 | 4 | 2.293 | 0.92x | 46% |
| qft | 22 | fixed-total-cores | 4 | 2 | 2.449 | 0.87x | 22% |
| qft | 22 | fixed-total-cores | 8 | 1 | 2.434 | 0.87x | 11% |
| qft | 22 | one-thread-per-rank | 1 | 1 | 5.454 | 1.00x | 100% |
| qft | 22 | one-thread-per-rank | 2 | 1 | 2.959 | 1.84x | 92% |
| qft | 22 | one-thread-per-rank | 4 | 1 | 2.237 | 2.44x | 61% |
| qft | 22 | one-thread-per-rank | 8 | 1 | 2.319 | 2.35x | 29% |

## Weak scaling (measured)

| circuit | thread policy | ranks | qubits | amplitudes/rank | wall (s) | efficiency |
|---|---|---:|---:|---:|---:|---:|
| ising | one-thread-per-rank | 1 | 20 | 1,048,576 | 0.416 | 100% |
| ising | one-thread-per-rank | 2 | 21 | 1,048,576 | 0.483 | 86% |
| ising | one-thread-per-rank | 4 | 22 | 1,048,576 | 0.715 | 58% |
| ising | one-thread-per-rank | 8 | 23 | 1,048,576 | 1.647 | 25% |

## Cost-model accuracy

- 42 of 42 distributed configurations sent exactly the number of bytes the cost model predicted.
