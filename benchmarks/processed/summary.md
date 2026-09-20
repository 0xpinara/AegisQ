## Measurement host

- host: `pias-MacBook-Pro.local`
- CPU: Apple M2 (8.0 logical cores)
- OS: macOS-15.6.1-arm64-arm-64bit
- compiler: AppleClang 17.0.0.17000603
- MPI: nan
- AegisQ 0.1.0 at commit `c980b47c70d0`

## Communication-aware placement (measured)

| circuit | qubits | ranks | baseline MPI bytes | optimized MPI bytes | reduction | baseline wall (s) | optimized wall (s) | wall change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ghz | 20 | 2 | 8,388,608 | 8,388,608 | 0.0% | 0.009 | 0.008 | -14.6% |
| ghz | 20 | 4 | 16,777,216 | 16,777,216 | 0.0% | 0.008 | 0.010 | +28.0% |
| ghz | 20 | 8 | 25,165,824 | 25,165,824 | 0.0% | 0.015 | 0.014 | -6.6% |
| grover | 20 | 2 | 536,870,912 | 150,994,944 | 71.9% | 0.468 | 0.440 | -6.0% |
| grover | 20 | 4 | 1,207,959,552 | 301,989,888 | 75.0% | 0.483 | 0.429 | -11.3% |
| grover | 20 | 8 | 1,879,048,192 | 520,093,696 | 72.3% | 0.547 | 0.479 | -12.5% |
| ising | 20 | 2 | 150,994,944 | 83,886,080 | 44.4% | 0.127 | 0.130 | +2.4% |
| ising | 20 | 4 | 301,989,888 | 234,881,024 | 22.2% | 0.132 | 0.132 | +0.0% |
| ising | 20 | 8 | 452,984,832 | 385,875,968 | 14.8% | 0.173 | 0.163 | -5.6% |
| qft | 20 | 2 | 343,932,928 | 25,165,824 | 92.7% | 0.349 | 0.327 | -6.5% |
| qft | 20 | 4 | 671,088,640 | 67,108,864 | 90.0% | 0.345 | 0.303 | -12.1% |
| qft | 20 | 8 | 981,467,136 | 125,829,120 | 87.2% | 0.383 | 0.260 | -32.1% |
| random | 20 | 2 | 159,383,552 | 83,886,080 | 47.4% | 0.108 | 0.104 | -3.5% |
| random | 20 | 4 | 251,658,240 | 167,772,160 | 33.3% | 0.111 | 0.101 | -8.7% |
| random | 20 | 8 | 411,041,792 | 251,658,240 | 38.8% | 0.150 | 0.134 | -10.8% |

## Strong scaling (measured)

| circuit | qubits | thread policy | ranks | threads/rank | wall (s) | speedup | efficiency |
|---|---:|---|---:|---:|---:|---:|---:|
| ising | 22 | one-thread-per-rank | 1 | 1 | 1.898 | 1.00x | 100% |
| ising | 22 | one-thread-per-rank | 2 | 1 | 1.027 | 1.85x | 92% |
| ising | 22 | one-thread-per-rank | 4 | 1 | 0.770 | 2.46x | 62% |
| ising | 22 | one-thread-per-rank | 8 | 1 | 0.772 | 2.46x | 31% |
| qft | 22 | one-thread-per-rank | 1 | 1 | 5.356 | 1.00x | 100% |
| qft | 22 | one-thread-per-rank | 2 | 1 | 3.016 | 1.78x | 89% |
| qft | 22 | one-thread-per-rank | 4 | 1 | 2.375 | 2.26x | 56% |
| qft | 22 | one-thread-per-rank | 8 | 1 | 2.435 | 2.20x | 27% |

## Weak scaling (measured)

| circuit | thread policy | ranks | qubits | amplitudes/rank | wall (s) | efficiency |
|---|---|---:|---:|---:|---:|---:|
| ising | one-thread-per-rank | 1 | 20 | 1,048,576 | 0.418 | 100% |
| ising | one-thread-per-rank | 2 | 21 | 1,048,576 | 0.487 | 86% |
| ising | one-thread-per-rank | 4 | 22 | 1,048,576 | 0.741 | 56% |
| ising | one-thread-per-rank | 8 | 23 | 1,048,576 | 1.657 | 25% |

## Placement search quality (measured)

| qubits | ranks | candidate sets | samples | needing the heuristic | optimum found | worst gap | median speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 4 | 153 | 33 | 31 | 33/33 | 0.00% | 3x |
| 18 | 8 | 816 | 33 | 31 | 33/33 | 0.00% | 11x |
| 18 | 16 | 3,060 | 33 | 31 | 33/33 | 0.00% | 40x |

## Local kernel bandwidth (measured)

| kernel | threads | GB/s | in-place reference | fraction |
|---|---:|---:|---:|---:|
| cx | 1 | 18.2 | 31.7 | 57% |
| cx | 2 | 33.8 | 59.7 | 57% |
| cx | 4 | 41.2 | 79.7 | 52% |
| cx | 8 | 38.0 | 70.8 | 54% |
| cz | 1 | 36.4 | 31.7 | 115% |
| cz | 2 | 41.1 | 59.7 | 69% |
| cz | 4 | 41.9 | 79.7 | 53% |
| cz | 8 | 45.3 | 70.8 | 64% |
| h | 1 | 19.9 | 31.7 | 63% |
| h | 2 | 39.5 | 59.7 | 66% |
| h | 4 | 71.8 | 79.7 | 90% |
| h | 8 | 76.4 | 70.8 | 108% |
| rz | 1 | 31.4 | 31.7 | 99% |
| rz | 2 | 60.1 | 59.7 | 101% |
| rz | 4 | 78.2 | 79.7 | 98% |
| rz | 8 | 77.8 | 70.8 | 110% |
| swap | 1 | 38.1 | 31.7 | 120% |
| swap | 2 | 41.1 | 59.7 | 69% |
| swap | 4 | 44.9 | 79.7 | 56% |
| swap | 8 | 42.5 | 70.8 | 60% |

## Post-quantum primitives (measured)

| algorithm | operation | median (us) | bytes |
|---|---|---:|---:|
| ML-DSA-44 | keygen | 32.9 | 1312 |
| ML-DSA-44 | sign | 60.8 | 2420 |
| ML-DSA-44 | verify | 30.9 | 2420 |
| ML-DSA-65 | keygen | 53.7 | 1952 |
| ML-DSA-65 | sign | 94.4 | 3309 |
| ML-DSA-65 | verify | 49.6 | 3309 |
| ML-DSA-87 | keygen | 89.1 | 2592 |
| ML-DSA-87 | sign | 139.7 | 4627 |
| ML-DSA-87 | verify | 83.3 | 4627 |
| ML-KEM-1024 | decapsulate | 27.7 | 32 |
| ML-KEM-1024 | encapsulate | 24.3 | 1568 |
| ML-KEM-1024 | keygen | 23.1 | 1568 |
| ML-KEM-512 | decapsulate | 14.6 | 32 |
| ML-KEM-512 | encapsulate | 12.9 | 768 |
| ML-KEM-512 | keygen | 12.4 | 800 |
| ML-KEM-768 | decapsulate | 20.0 | 32 |
| ML-KEM-768 | encapsulate | 17.7 | 1088 |
| ML-KEM-768 | keygen | 17.0 | 1184 |

## Secure job envelope (measured)

| step | median (us) | bytes |
|---|---:|---:|
| bundle_size | - | 78059 |
| envelope_fixed_overhead | - | 7083 |
| pack_job | 3131 | 78059 |
| payload_base64 | - | 70976 |
| payload_plaintext | - | 53232 |
| verify_and_open | 6079 | 78059 |

## Optimisation levers (measured)

| circuit | ranks | baseline (MiB) | fusion | static placement | windowed placement | placement + fusion |
|---|---:|---:|---:|---:|---:|---:|
| ghz | 2 | 8 | 0.0% | 0.0% | 0.0% | 0.0% |
| ghz | 4 | 16 | 0.0% | 0.0% | 0.0% | 0.0% |
| ghz | 8 | 24 | 0.0% | 0.0% | 0.0% | 0.0% |
| grover | 2 | 512 | 18.8% | 71.9% | 85.9% | 84.4% |
| grover | 4 | 1152 | 16.7% | 75.0% | 87.5% | 86.1% |
| grover | 8 | 1792 | 16.1% | 72.3% | 87.1% | 86.6% |
| ising | 2 | 144 | 0.0% | 44.4% | 44.4% | 44.4% |
| ising | 4 | 288 | 0.0% | 22.2% | 25.0% | 22.2% |
| ising | 8 | 432 | 0.0% | 14.8% | 16.7% | 14.8% |
| qft | 2 | 328 | 0.0% | 92.7% | 95.1% | 92.7% |
| qft | 4 | 640 | 0.0% | 90.0% | 96.2% | 90.0% |
| qft | 8 | 936 | 0.0% | 87.2% | 95.7% | 87.2% |
| random | 2 | 152 | 0.0% | 47.4% | 57.9% | 68.4% |
| random | 4 | 240 | 0.0% | 33.3% | 40.0% | 56.7% |
| random | 8 | 392 | 8.2% | 38.8% | 42.9% | 53.1% |

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
