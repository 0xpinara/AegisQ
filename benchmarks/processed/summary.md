## Measurement host

- host: `pias-MacBook-Pro.local`
- CPU: Apple M2 (8 logical cores)
- OS: macOS-15.6.1-arm64-arm-64bit
- compiler: AppleClang 17.0.0.17000603
- MPI: Open MPI v5.0.10, package: Open MPI brew@Sequoia-arm64.local Distribution, ident: 5.0.10, repo rev: v5.0.10, Feb 23, 2026
- AegisQ 0.1.0 at commit `b67b6064b543`

## Communication-aware placement (measured)

| circuit | qubits | ranks | baseline MPI bytes | optimized MPI bytes | reduction | baseline wall (s) | optimized wall (s) | wall change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ghz | 20 | 2 | 8,388,608 | 8,388,608 | 0.0% | 0.008 | 0.009 | +12.9% |
| ghz | 20 | 4 | 16,777,216 | 16,777,216 | 0.0% | 0.012 | 0.010 | -17.9% |
| ghz | 20 | 8 | 25,165,824 | 25,165,824 | 0.0% | 0.014 | 0.017 | +20.7% |
| grover | 20 | 2 | 536,870,912 | 150,994,944 | 71.9% | 0.461 | 0.447 | -3.1% |
| grover | 20 | 4 | 1,207,959,552 | 301,989,888 | 75.0% | 0.485 | 0.420 | -13.3% |
| grover | 20 | 8 | 1,879,048,192 | 520,093,696 | 72.3% | 0.569 | 0.468 | -17.8% |
| ising | 20 | 2 | 150,994,944 | 83,886,080 | 44.4% | 0.131 | 0.126 | -3.7% |
| ising | 20 | 4 | 301,989,888 | 234,881,024 | 22.2% | 0.136 | 0.126 | -7.7% |
| ising | 20 | 8 | 452,984,832 | 385,875,968 | 14.8% | 0.168 | 0.198 | +17.7% |
| qft | 20 | 2 | 343,932,928 | 25,165,824 | 92.7% | 0.359 | 0.326 | -9.1% |
| qft | 20 | 4 | 671,088,640 | 67,108,864 | 90.0% | 0.362 | 0.329 | -9.2% |
| qft | 20 | 8 | 981,467,136 | 125,829,120 | 87.2% | 0.374 | 0.296 | -20.8% |
| random | 20 | 2 | 159,383,552 | 83,886,080 | 47.4% | 0.108 | 0.106 | -2.5% |
| random | 20 | 4 | 251,658,240 | 167,772,160 | 33.3% | 0.113 | 0.102 | -8.9% |
| random | 20 | 8 | 411,041,792 | 251,658,240 | 38.8% | 0.156 | 0.141 | -9.6% |

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

## Post-quantum primitives (measured)

| algorithm | operation | median (us) | bytes |
|---|---|---:|---:|
| ML-DSA-44 | keygen | 32.8 | 1312 |
| ML-DSA-44 | sign | 61.5 | 2420 |
| ML-DSA-44 | verify | 31.2 | 2420 |
| ML-DSA-65 | keygen | 53.8 | 1952 |
| ML-DSA-65 | sign | 101.8 | 3309 |
| ML-DSA-65 | verify | 49.6 | 3309 |
| ML-DSA-87 | keygen | 91.8 | 2592 |
| ML-DSA-87 | sign | 138.2 | 4627 |
| ML-DSA-87 | verify | 83.4 | 4627 |
| ML-KEM-1024 | decapsulate | 27.7 | 32 |
| ML-KEM-1024 | encapsulate | 24.1 | 1568 |
| ML-KEM-1024 | keygen | 23.2 | 1568 |
| ML-KEM-512 | decapsulate | 14.5 | 32 |
| ML-KEM-512 | encapsulate | 12.9 | 768 |
| ML-KEM-512 | keygen | 12.1 | 800 |
| ML-KEM-768 | decapsulate | 20.0 | 32 |
| ML-KEM-768 | encapsulate | 17.5 | 1088 |
| ML-KEM-768 | keygen | 16.9 | 1184 |

## Secure job envelope (measured)

| step | median (us) | bytes |
|---|---:|---:|
| bundle_size | - | 78059 |
| envelope_fixed_overhead | - | 7083 |
| pack_job | 3089 | 78059 |
| payload_base64 | - | 70976 |
| payload_plaintext | - | 53232 |
| verify_and_open | 5571 | 78059 |

## Optimisation levers, separately and together (measured)

| circuit | ranks | baseline (MiB) | fusion only | placement only | both |
|---|---:|---:|---:|---:|---:|
| ghz | 2 | 8 | 0.0% | 0.0% | 0.0% |
| ghz | 4 | 16 | 0.0% | 0.0% | 0.0% |
| ghz | 8 | 24 | 0.0% | 0.0% | 0.0% |
| grover | 2 | 512 | 18.8% | 71.9% | 84.4% |
| grover | 4 | 1152 | 16.7% | 75.0% | 86.1% |
| grover | 8 | 1792 | 16.1% | 72.3% | 86.6% |
| ising | 2 | 144 | 0.0% | 44.4% | 44.4% |
| ising | 4 | 288 | 0.0% | 22.2% | 22.2% |
| ising | 8 | 432 | 0.0% | 14.8% | 14.8% |
| qft | 2 | 328 | 0.0% | 92.7% | 92.7% |
| qft | 4 | 640 | 0.0% | 90.0% | 90.0% |
| qft | 8 | 936 | 0.0% | 87.2% | 87.2% |
| random | 2 | 152 | 0.0% | 47.4% | 68.4% |
| random | 4 | 240 | 0.0% | 33.3% | 56.7% |
| random | 8 | 392 | 8.2% | 38.8% | 53.1% |

## Grover query scaling (measured)

| search space | qubits | classical expected queries | Grover queries | measured success |
|---:|---:|---:|---:|---:|
| 4 | 2 | 2.5 | 1 | 100.0% |
| 8 | 4 | 4.5 | 2 | 95.4% |
| 16 | 6 | 8.5 | 3 | 96.6% |
| 32 | 8 | 16.5 | 4 | 99.9% |
| 64 | 10 | 32.5 | 6 | 99.7% |
| 128 | 12 | 64.5 | 8 | 99.6% |
| 256 | 14 | 128.5 | 12 | 100.0% |

## Cost-model accuracy

- 72 of 72 distributed configurations sent exactly the number of bytes the cost model predicted.
