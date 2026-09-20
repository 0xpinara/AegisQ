## Measurement host

- host: `pias-MacBook-Pro.local`
- CPU: Apple M2 (8 logical cores)
- OS: macOS-15.6.1-arm64-arm-64bit
- compiler: AppleClang 17.0.0.17000603
- MPI: Open MPI v5.0.10, package: Open MPI brew@Sequoia-arm64.local Distribution, ident: 5.0.10, repo rev: v5.0.10, Feb 23, 2026
- AegisQ 0.1.0 at commit `67fb99bd336c`

## Communication-aware placement (measured)

| circuit | qubits | ranks | baseline MPI bytes | optimized MPI bytes | reduction | baseline wall (s) | optimized wall (s) | wall change | resolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| ghz | 20 | 2 | 8,388,608 | 8,388,608 | 0.0% | 0.011 | 0.008 | -28.5% | no |
| ghz | 20 | 4 | 16,777,216 | 16,777,216 | 0.0% | 0.010 | 0.009 | -4.1% | no |
| ghz | 20 | 8 | 25,165,824 | 25,165,824 | 0.0% | 0.017 | 0.019 | +14.8% | no |
| grover | 20 | 2 | 536,870,912 | 150,994,944 | 71.9% | 0.466 | 0.445 | -4.4% | no |
| grover | 20 | 4 | 1,207,959,552 | 301,989,888 | 75.0% | 0.515 | 0.461 | -10.6% | yes |
| grover | 20 | 8 | 1,879,048,192 | 520,093,696 | 72.3% | 0.580 | 0.478 | -17.7% | no |
| ising | 20 | 2 | 150,994,944 | 83,886,080 | 44.4% | 0.123 | 0.175 | +42.5% | yes |
| ising | 20 | 4 | 301,989,888 | 234,881,024 | 22.2% | 0.115 | 0.109 | -5.1% | yes |
| ising | 20 | 8 | 452,984,832 | 385,875,968 | 14.8% | 0.158 | 0.152 | -3.8% | no |
| qft | 20 | 2 | 343,932,928 | 25,165,824 | 92.7% | 0.362 | 0.325 | -10.2% | no |
| qft | 20 | 4 | 671,088,640 | 67,108,864 | 90.0% | 0.357 | 0.311 | -12.8% | yes |
| qft | 20 | 8 | 981,467,136 | 125,829,120 | 87.2% | 0.361 | 0.265 | -26.6% | yes |
| random | 20 | 2 | 159,383,552 | 83,886,080 | 47.4% | 0.091 | 0.116 | +27.7% | no |
| random | 20 | 4 | 251,658,240 | 167,772,160 | 33.3% | 0.109 | 0.100 | -8.4% | yes |
| random | 20 | 8 | 411,041,792 | 251,658,240 | 38.8% | 0.138 | 0.140 | +1.8% | no |

A wall-time change is *resolved* only if it exceeds the noise floor for its rank count -- the largest change measured on circuits the optimiser leaves byte-for-byte unchanged, whose true effect is therefore zero -- and its repeat range does not overlap the baseline's. Byte counts are exact.

## Strong scaling (measured)

| circuit | qubits | thread policy | ranks | threads/rank | wall (s) | speedup | efficiency |
|---|---:|---|---:|---:|---:|---:|---:|
| ising | 22 | one-thread-per-rank | 1 | 1 | 1.909 | 1.00x | 100% |
| ising | 22 | one-thread-per-rank | 2 | 1 | 1.045 | 1.83x | 91% |
| ising | 22 | one-thread-per-rank | 4 | 1 | 0.759 | 2.52x | 63% |
| ising | 22 | one-thread-per-rank | 8 | 1 | 0.735 | 2.60x | 32% |
| qft | 22 | one-thread-per-rank | 1 | 1 | 5.433 | 1.00x | 100% |
| qft | 22 | one-thread-per-rank | 2 | 1 | 3.054 | 1.78x | 89% |
| qft | 22 | one-thread-per-rank | 4 | 1 | 2.378 | 2.28x | 57% |
| qft | 22 | one-thread-per-rank | 8 | 1 | 2.416 | 2.25x | 28% |

## Weak scaling (measured)

| circuit | thread policy | ranks | qubits | amplitudes/rank | wall (s) | efficiency |
|---|---|---:|---:|---:|---:|---:|
| ising | one-thread-per-rank | 1 | 20 | 1,048,576 | 0.399 | 100% |
| ising | one-thread-per-rank | 2 | 21 | 1,048,576 | 0.501 | 79% |
| ising | one-thread-per-rank | 4 | 22 | 1,048,576 | 0.624 | 64% |
| ising | one-thread-per-rank | 8 | 23 | 1,048,576 | 1.592 | 25% |

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
| 18 | 16 | 3,060 | 33 | 31 | 33/33 | 0.00% | 40x |

## Local kernel bandwidth (measured)

| kernel | threads | GB/s | in-place reference | fraction |
|---|---:|---:|---:|---:|
| cx | 1 | 17.0 | 31.4 | 54% |
| cx | 2 | 34.0 | 58.8 | 58% |
| cx | 4 | 41.0 | 80.3 | 51% |
| cx | 8 | 41.2 | 79.6 | 52% |
| cz | 1 | 35.4 | 31.4 | 113% |
| cz | 2 | 41.5 | 58.8 | 71% |
| cz | 4 | 41.3 | 80.3 | 51% |
| cz | 8 | 49.0 | 79.6 | 62% |
| h | 1 | 20.1 | 31.4 | 64% |
| h | 2 | 38.3 | 58.8 | 65% |
| h | 4 | 75.0 | 80.3 | 93% |
| h | 8 | 78.2 | 79.6 | 98% |
| rz | 1 | 31.6 | 31.4 | 101% |
| rz | 2 | 59.9 | 58.8 | 102% |
| rz | 4 | 80.1 | 80.3 | 100% |
| rz | 8 | 80.1 | 79.6 | 101% |
| swap | 1 | 35.4 | 31.4 | 113% |
| swap | 2 | 41.7 | 58.8 | 71% |
| swap | 4 | 44.0 | 80.3 | 55% |
| swap | 8 | 44.5 | 79.6 | 56% |

## Post-quantum primitives (measured)

| algorithm | operation | median (us) | bytes |
|---|---|---:|---:|
| ML-DSA-44 | keygen | 32.8 | 1312 |
| ML-DSA-44 | sign | 59.5 | 2420 |
| ML-DSA-44 | verify | 30.8 | 2420 |
| ML-DSA-65 | keygen | 53.6 | 1952 |
| ML-DSA-65 | sign | 93.0 | 3309 |
| ML-DSA-65 | verify | 49.3 | 3309 |
| ML-DSA-87 | keygen | 89.0 | 2592 |
| ML-DSA-87 | sign | 134.0 | 4627 |
| ML-DSA-87 | verify | 83.2 | 4627 |
| ML-KEM-1024 | decapsulate | 27.7 | 32 |
| ML-KEM-1024 | encapsulate | 24.4 | 1568 |
| ML-KEM-1024 | keygen | 23.5 | 1568 |
| ML-KEM-512 | decapsulate | 14.5 | 32 |
| ML-KEM-512 | encapsulate | 12.9 | 768 |
| ML-KEM-512 | keygen | 12.3 | 800 |
| ML-KEM-768 | decapsulate | 21.0 | 32 |
| ML-KEM-768 | encapsulate | 18.3 | 1088 |
| ML-KEM-768 | keygen | 16.9 | 1184 |

## Secure job envelope (measured)

| step | median (us) | bytes |
|---|---:|---:|
| bundle_size | - | 78059 |
| envelope_fixed_overhead | - | 7083 |
| pack_job | 3096 | 78059 |
| payload_base64 | - | 70976 |
| payload_plaintext | - | 53232 |
| verify_and_open | 5788 | 78059 |

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
