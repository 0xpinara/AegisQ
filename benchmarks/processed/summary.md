## Measurement host

- host: `pias-MacBook-Pro.local`
- CPU: Apple M2 (8 logical cores)
- OS: macOS-15.6.1-arm64-arm-64bit
- compiler: AppleClang 17.0.0.17000603
- MPI: Open MPI v5.0.10, package: Open MPI brew@Sequoia-arm64.local Distribution, ident: 5.0.10, repo rev: v5.0.10, Feb 23, 2026
- AegisQ 0.1.0 at commit `1a8b8b3ff33e`

## Communication-aware placement (measured)

| circuit | qubits | ranks | baseline MPI bytes | optimized MPI bytes | reduction | baseline wall (s) | optimized wall (s) | wall change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ghz | 20 | 2 | 8,388,608 | 8,388,608 | 0.0% | 0.009 | 0.009 | +0.3% |
| ghz | 20 | 4 | 16,777,216 | 16,777,216 | 0.0% | 0.007 | 0.010 | +32.4% |
| ghz | 20 | 8 | 25,165,824 | 25,165,824 | 0.0% | 0.014 | 0.014 | +1.7% |
| grover | 20 | 2 | 536,870,912 | 150,994,944 | 71.9% | 0.466 | 0.448 | -3.8% |
| grover | 20 | 4 | 1,207,959,552 | 301,989,888 | 75.0% | 0.515 | 0.453 | -12.1% |
| grover | 20 | 8 | 1,879,048,192 | 520,093,696 | 72.3% | 0.595 | 0.454 | -23.7% |
| ising | 20 | 2 | 150,994,944 | 83,886,080 | 44.4% | 0.130 | 0.132 | +1.0% |
| ising | 20 | 4 | 301,989,888 | 234,881,024 | 22.2% | 0.112 | 0.109 | -2.9% |
| ising | 20 | 8 | 452,984,832 | 385,875,968 | 14.8% | 0.157 | 0.154 | -1.6% |
| qft | 20 | 2 | 343,932,928 | 25,165,824 | 92.7% | 0.373 | 0.322 | -13.6% |
| qft | 20 | 4 | 671,088,640 | 67,108,864 | 90.0% | 0.293 | 0.270 | -7.9% |
| qft | 20 | 8 | 981,467,136 | 125,829,120 | 87.2% | 0.342 | 0.285 | -16.5% |
| random | 20 | 2 | 159,383,552 | 83,886,080 | 47.4% | 0.093 | 0.087 | -7.0% |
| random | 20 | 4 | 251,658,240 | 167,772,160 | 33.3% | 0.119 | 0.101 | -15.6% |
| random | 20 | 8 | 411,041,792 | 251,658,240 | 38.8% | 0.155 | 0.121 | -22.0% |

## Strong scaling (measured)

| circuit | qubits | thread policy | ranks | threads/rank | wall (s) | speedup | efficiency |
|---|---:|---|---:|---:|---:|---:|---:|
| ising | 22 | one-thread-per-rank | 1 | 1 | 1.887 | 1.00x | 100% |
| ising | 22 | one-thread-per-rank | 2 | 1 | 1.014 | 1.86x | 93% |
| ising | 22 | one-thread-per-rank | 4 | 1 | 0.634 | 2.98x | 74% |
| ising | 22 | one-thread-per-rank | 8 | 1 | 0.667 | 2.83x | 35% |
| qft | 22 | one-thread-per-rank | 1 | 1 | 5.002 | 1.00x | 100% |
| qft | 22 | one-thread-per-rank | 2 | 1 | 2.713 | 1.84x | 92% |
| qft | 22 | one-thread-per-rank | 4 | 1 | 1.870 | 2.67x | 67% |
| qft | 22 | one-thread-per-rank | 8 | 1 | 1.918 | 2.61x | 33% |

## Weak scaling (measured)

| circuit | thread policy | ranks | qubits | amplitudes/rank | wall (s) | efficiency |
|---|---|---:|---:|---:|---:|---:|
| ising | one-thread-per-rank | 1 | 20 | 1,048,576 | 0.398 | 100% |
| ising | one-thread-per-rank | 2 | 21 | 1,048,576 | 0.445 | 89% |
| ising | one-thread-per-rank | 4 | 22 | 1,048,576 | 0.668 | 60% |
| ising | one-thread-per-rank | 8 | 23 | 1,048,576 | 1.488 | 27% |

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
| 18 | 16 | 3,060 | 33 | 31 | 33/33 | 0.00% | 39x |

## Local kernel bandwidth (measured)

| kernel | threads | GB/s | in-place reference | fraction |
|---|---:|---:|---:|---:|
| cx | 1 | 17.9 | 31.5 | 57% |
| cx | 2 | 34.3 | 59.9 | 57% |
| cx | 4 | 42.2 | 81.0 | 52% |
| cx | 8 | 39.7 | 73.0 | 54% |
| cz | 1 | 34.2 | 31.5 | 109% |
| cz | 2 | 42.1 | 59.9 | 70% |
| cz | 4 | 44.1 | 81.0 | 54% |
| cz | 8 | 44.0 | 73.0 | 60% |
| h | 1 | 20.4 | 31.5 | 65% |
| h | 2 | 39.1 | 59.9 | 65% |
| h | 4 | 74.6 | 81.0 | 92% |
| h | 8 | 77.2 | 73.0 | 106% |
| rz | 1 | 32.0 | 31.5 | 102% |
| rz | 2 | 61.5 | 59.9 | 103% |
| rz | 4 | 81.9 | 81.0 | 101% |
| rz | 8 | 81.8 | 73.0 | 112% |
| swap | 1 | 36.0 | 31.5 | 114% |
| swap | 2 | 42.6 | 59.9 | 71% |
| swap | 4 | 45.6 | 81.0 | 56% |
| swap | 8 | 43.9 | 73.0 | 60% |

## Post-quantum primitives (measured)

| algorithm | operation | median (us) | bytes |
|---|---|---:|---:|
| ML-DSA-44 | keygen | 33.0 | 1312 |
| ML-DSA-44 | sign | 63.0 | 2420 |
| ML-DSA-44 | verify | 31.3 | 2420 |
| ML-DSA-65 | keygen | 53.8 | 1952 |
| ML-DSA-65 | sign | 91.0 | 3309 |
| ML-DSA-65 | verify | 50.7 | 3309 |
| ML-DSA-87 | keygen | 89.4 | 2592 |
| ML-DSA-87 | sign | 139.5 | 4627 |
| ML-DSA-87 | verify | 83.4 | 4627 |
| ML-KEM-1024 | decapsulate | 29.4 | 32 |
| ML-KEM-1024 | encapsulate | 25.7 | 1568 |
| ML-KEM-1024 | keygen | 24.5 | 1568 |
| ML-KEM-512 | decapsulate | 14.5 | 32 |
| ML-KEM-512 | encapsulate | 12.8 | 768 |
| ML-KEM-512 | keygen | 12.1 | 800 |
| ML-KEM-768 | decapsulate | 21.2 | 32 |
| ML-KEM-768 | encapsulate | 18.6 | 1088 |
| ML-KEM-768 | keygen | 16.8 | 1184 |

## Secure job envelope (measured)

| step | median (us) | bytes |
|---|---:|---:|
| bundle_size | - | 78059 |
| envelope_fixed_overhead | - | 7083 |
| pack_job | 3179 | 78059 |
| payload_base64 | - | 70976 |
| payload_plaintext | - | 53232 |
| verify_and_open | 5707 | 78059 |

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
