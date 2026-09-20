## Measurement host

- host: `pias-MacBook-Pro.local`
- CPU: Apple M2 (8 logical cores)
- OS: macOS-15.6.1-arm64-arm-64bit
- compiler: AppleClang 17.0.0.17000603
- MPI: Open MPI v5.0.10, package: Open MPI brew@Sequoia-arm64.local Distribution, ident: 5.0.10, repo rev: v5.0.10, Feb 23, 2026
- AegisQ 0.1.0 at commit `1c782e4424ad`

## Communication-aware placement (measured)

| circuit | qubits | ranks | baseline MPI bytes | optimized MPI bytes | reduction | baseline wall (s) | optimized wall (s) | wall change | resolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| ghz | 20 | 2 | 8,388,608 | 8,388,608 | 0.0% | 0.007 | 0.008 | +5.7% | no |
| ghz | 20 | 4 | 16,777,216 | 16,777,216 | 0.0% | 0.008 | 0.008 | -4.6% | no |
| ghz | 20 | 8 | 25,165,824 | 25,165,824 | 0.0% | 0.012 | 0.013 | +4.5% | no |
| grover | 20 | 2 | 536,870,912 | 150,994,944 | 71.9% | 0.425 | 0.404 | -4.9% | no |
| grover | 20 | 4 | 1,207,959,552 | 301,989,888 | 75.0% | 0.417 | 0.388 | -7.0% | no |
| grover | 20 | 8 | 1,879,048,192 | 520,093,696 | 72.3% | 0.524 | 0.416 | -20.5% | yes |
| ising | 20 | 2 | 150,994,944 | 83,886,080 | 44.4% | 0.115 | 0.113 | -1.1% | no |
| ising | 20 | 4 | 301,989,888 | 234,881,024 | 22.2% | 0.128 | 0.127 | -0.9% | no |
| ising | 20 | 8 | 452,984,832 | 385,875,968 | 14.8% | 0.170 | 0.165 | -2.9% | no |
| qft | 20 | 2 | 343,932,928 | 25,165,824 | 92.7% | 0.296 | 0.271 | -8.5% | yes |
| qft | 20 | 4 | 671,088,640 | 67,108,864 | 90.0% | 0.306 | 0.281 | -8.2% | no |
| qft | 20 | 8 | 981,467,136 | 125,829,120 | 87.2% | 0.309 | 0.238 | -22.8% | no |
| random | 20 | 2 | 159,383,552 | 83,886,080 | 47.4% | 0.098 | 0.086 | -12.4% | no |
| random | 20 | 4 | 251,658,240 | 167,772,160 | 33.3% | 0.105 | 0.101 | -4.3% | no |
| random | 20 | 8 | 411,041,792 | 251,658,240 | 38.8% | 0.141 | 0.131 | -7.1% | no |

A wall-time change is *resolved* only if it exceeds the noise floor for its rank count -- the largest change measured on circuits the optimiser leaves byte-for-byte unchanged, whose true effect is therefore zero -- and its repeat range does not overlap the baseline's. Byte counts are exact.

## Strong scaling (measured)

| circuit | qubits | thread policy | ranks | threads/rank | wall (s) | speedup | efficiency |
|---|---:|---|---:|---:|---:|---:|---:|
| ising | 22 | one-thread-per-rank | 1 | 1 | 1.824 | 1.00x | 100% |
| ising | 22 | one-thread-per-rank | 2 | 1 | 0.999 | 1.83x | 91% |
| ising | 22 | one-thread-per-rank | 4 | 1 | 0.611 | 2.98x | 75% |
| ising | 22 | one-thread-per-rank | 8 | 1 | 0.618 | 2.95x | 37% |
| qft | 22 | one-thread-per-rank | 1 | 1 | 5.293 | 1.00x | 100% |
| qft | 22 | one-thread-per-rank | 2 | 1 | 2.782 | 1.90x | 95% |
| qft | 22 | one-thread-per-rank | 4 | 1 | 1.947 | 2.72x | 68% |
| qft | 22 | one-thread-per-rank | 8 | 1 | 1.907 | 2.78x | 35% |

## Weak scaling (measured)

| circuit | thread policy | ranks | qubits | amplitudes/rank | wall (s) | efficiency |
|---|---|---:|---:|---:|---:|---:|
| ising | one-thread-per-rank | 1 | 20 | 1,048,576 | 0.387 | 100% |
| ising | one-thread-per-rank | 2 | 21 | 1,048,576 | 0.448 | 86% |
| ising | one-thread-per-rank | 4 | 22 | 1,048,576 | 0.604 | 64% |
| ising | one-thread-per-rank | 8 | 23 | 1,048,576 | 1.504 | 26% |

## Single-precision error (measured)

| circuit | qubits | gates | worst 1-fidelity | worst amplitude error |
|---|---:|---:|---:|---:|
| grover | 18 | 1082 | 3.39e-13 | 1.54e-06 |
| ising | 18 | 294 | 2.61e-13 | 7.08e-08 |
| qft | 18 | 792 | 3.20e-13 | 1.22e-08 |
| random | 18 | 43 | 1.29e-14 | 7.53e-08 |
| random | 18 | 164 | 8.12e-14 | 6.66e-08 |
| random | 18 | 639 | 3.49e-13 | 9.92e-09 |
| random | 18 | 643 | 3.34e-13 | 9.17e-09 |
| random | 18 | 654 | 3.08e-13 | 7.47e-09 |
| random | 18 | 656 | 3.32e-13 | 7.22e-09 |
| random | 18 | 2510 | 1.39e-12 | 3.08e-08 |
| random | 18 | 2527 | 1.35e-12 | 2.98e-08 |
| random | 18 | 2533 | 1.35e-12 | 3.43e-08 |
| random | 18 | 2553 | 1.46e-12 | 3.38e-08 |
| random | 18 | 2577 | 1.37e-12 | 3.20e-08 |
| random | 18 | 10141 | 5.63e-12 | 1.16e-07 |
| random | 18 | 10151 | 5.65e-12 | 1.14e-07 |
| random | 18 | 10159 | 5.48e-12 | 1.15e-07 |
| random | 18 | 10198 | 5.72e-12 | 1.31e-07 |

## Placement search quality (measured)

| qubits | ranks | candidate sets | samples | needing the heuristic | optimum found | worst gap | median speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 4 | 153 | 33 | 31 | 33/33 | 0.00% | 3x |
| 18 | 8 | 816 | 33 | 31 | 33/33 | 0.00% | 12x |
| 18 | 16 | 3,060 | 33 | 31 | 33/33 | 0.00% | 41x |

## Local kernel bandwidth (measured)

| kernel | threads | GB/s | in-place reference | fraction |
|---|---:|---:|---:|---:|
| cx | 1 | 17.9 | 30.7 | 58% |
| cx | 2 | 32.8 | 59.2 | 55% |
| cx | 4 | 41.8 | 76.6 | 55% |
| cx | 8 | 42.0 | 80.8 | 52% |
| cz | 1 | 35.4 | 30.7 | 115% |
| cz | 2 | 41.2 | 59.2 | 70% |
| cz | 4 | 43.0 | 76.6 | 56% |
| cz | 8 | 46.4 | 80.8 | 57% |
| h | 1 | 20.1 | 30.7 | 65% |
| h | 2 | 39.5 | 59.2 | 67% |
| h | 4 | 73.7 | 76.6 | 96% |
| h | 8 | 77.9 | 80.8 | 96% |
| rz | 1 | 32.2 | 30.7 | 105% |
| rz | 2 | 60.7 | 59.2 | 103% |
| rz | 4 | 80.7 | 76.6 | 105% |
| rz | 8 | 84.5 | 80.8 | 105% |
| swap | 1 | 38.4 | 30.7 | 125% |
| swap | 2 | 40.4 | 59.2 | 68% |
| swap | 4 | 45.7 | 76.6 | 60% |
| swap | 8 | 43.3 | 80.8 | 54% |

## Post-quantum primitives (measured)

| algorithm | operation | median (us) | bytes |
|---|---|---:|---:|
| ML-DSA-44 | keygen | 33.6 | 1312 |
| ML-DSA-44 | sign | 63.1 | 2420 |
| ML-DSA-44 | verify | 30.9 | 2420 |
| ML-DSA-65 | keygen | 54.9 | 1952 |
| ML-DSA-65 | sign | 104.8 | 3309 |
| ML-DSA-65 | verify | 49.6 | 3309 |
| ML-DSA-87 | keygen | 93.9 | 2592 |
| ML-DSA-87 | sign | 136.3 | 4627 |
| ML-DSA-87 | verify | 83.5 | 4627 |
| ML-KEM-1024 | decapsulate | 27.9 | 32 |
| ML-KEM-1024 | encapsulate | 24.5 | 1568 |
| ML-KEM-1024 | keygen | 23.8 | 1568 |
| ML-KEM-512 | decapsulate | 15.3 | 32 |
| ML-KEM-512 | encapsulate | 13.3 | 768 |
| ML-KEM-512 | keygen | 12.9 | 800 |
| ML-KEM-768 | decapsulate | 20.5 | 32 |
| ML-KEM-768 | encapsulate | 18.0 | 1088 |
| ML-KEM-768 | keygen | 18.0 | 1184 |

## Secure job envelope (measured)

| step | median (us) | bytes |
|---|---:|---:|
| bundle_size | - | 78059 |
| envelope_fixed_overhead | - | 7083 |
| pack_job | 3174 | 78059 |
| payload_base64 | - | 70976 |
| payload_plaintext | - | 53232 |
| verify_and_open | 5683 | 78059 |

## Optimisation levers (measured)

| circuit | ranks | baseline (MiB) | fusion | static placement | windowed placement | placement + fusion | windowed + fusion |
|---|---:|---:|---:|---:|---:|---:|---:|
| ghz | 2 | 8 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| ghz | 4 | 16 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| ghz | 8 | 24 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| grover | 2 | 512 | 18.8% | 71.9% | 85.9% | 84.4% | 85.9% |
| grover | 4 | 1152 | 16.7% | 75.0% | 87.5% | 86.1% | 87.5% |
| grover | 8 | 1792 | 16.1% | 72.3% | 87.1% | 86.6% | 87.5% |
| ising | 2 | 144 | 0.0% | 44.4% | 44.4% | 44.4% | 44.4% |
| ising | 4 | 288 | 0.0% | 22.2% | 25.0% | 22.2% | 36.1% |
| ising | 8 | 432 | 0.0% | 14.8% | 16.7% | 14.8% | 33.3% |
| qft | 2 | 328 | 0.0% | 92.7% | 95.1% | 92.7% | 95.1% |
| qft | 4 | 640 | 0.0% | 90.0% | 96.2% | 90.0% | 96.2% |
| qft | 8 | 936 | 0.0% | 87.2% | 95.7% | 87.2% | 95.7% |
| random | 2 | 152 | 0.0% | 47.4% | 57.9% | 68.4% | 68.4% |
| random | 4 | 240 | 0.0% | 33.3% | 40.0% | 56.7% | 56.7% |
| random | 8 | 392 | 8.2% | 38.8% | 42.9% | 53.1% | 55.1% |

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

- 99 of 99 distributed configurations sent exactly the number of bytes the cost model predicted.
