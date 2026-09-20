## Measurement host

- host: `pias-MacBook-Pro.local`
- CPU: Apple M2 (8 logical cores)
- OS: macOS-15.6.1-arm64-arm-64bit
- compiler: AppleClang 17.0.0.17000603
- MPI: Open MPI v5.0.10, package: Open MPI brew@Sequoia-arm64.local Distribution, ident: 5.0.10, repo rev: v5.0.10, Feb 23, 2026
- AegisQ 0.1.0 at commit `5e2143839716`

## Communication-aware placement (measured)

| circuit | qubits | ranks | baseline MPI bytes | optimized MPI bytes | reduction | baseline wall (s) | optimized wall (s) | wall change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ghz | 20 | 2 | 8,388,608 | 8,388,608 | 0.0% | 0.008 | 0.008 | -6.2% |
| ghz | 20 | 4 | 16,777,216 | 16,777,216 | 0.0% | 0.009 | 0.008 | -7.7% |
| ghz | 20 | 8 | 25,165,824 | 25,165,824 | 0.0% | 0.014 | 0.015 | +11.0% |
| grover | 20 | 2 | 536,870,912 | 150,994,944 | 71.9% | 0.458 | 0.445 | -2.9% |
| grover | 20 | 4 | 1,207,959,552 | 301,989,888 | 75.0% | 0.492 | 0.442 | -10.2% |
| grover | 20 | 8 | 1,879,048,192 | 520,093,696 | 72.3% | 0.579 | 0.452 | -21.9% |
| ising | 20 | 2 | 150,994,944 | 83,886,080 | 44.4% | 0.128 | 0.124 | -3.4% |
| ising | 20 | 4 | 301,989,888 | 234,881,024 | 22.2% | 0.132 | 0.125 | -5.3% |
| ising | 20 | 8 | 452,984,832 | 385,875,968 | 14.8% | 0.208 | 0.177 | -14.6% |
| qft | 20 | 2 | 343,932,928 | 25,165,824 | 92.7% | 0.346 | 0.316 | -8.6% |
| qft | 20 | 4 | 671,088,640 | 67,108,864 | 90.0% | 0.364 | 0.303 | -16.6% |
| qft | 20 | 8 | 981,467,136 | 125,829,120 | 87.2% | 0.387 | 0.271 | -30.0% |
| random | 20 | 2 | 159,383,552 | 83,886,080 | 47.4% | 0.108 | 0.104 | -3.2% |
| random | 20 | 4 | 251,658,240 | 167,772,160 | 33.3% | 0.108 | 0.101 | -6.6% |
| random | 20 | 8 | 411,041,792 | 251,658,240 | 38.8% | 0.173 | 0.136 | -21.3% |

## Strong scaling (measured)

| circuit | qubits | thread policy | ranks | threads/rank | wall (s) | speedup | efficiency |
|---|---:|---|---:|---:|---:|---:|---:|
| ising | 22 | one-thread-per-rank | 1 | 1 | 1.900 | 1.00x | 100% |
| ising | 22 | one-thread-per-rank | 2 | 1 | 1.035 | 1.84x | 92% |
| ising | 22 | one-thread-per-rank | 4 | 1 | 0.746 | 2.55x | 64% |
| ising | 22 | one-thread-per-rank | 8 | 1 | 0.784 | 2.42x | 30% |
| qft | 22 | one-thread-per-rank | 1 | 1 | 5.391 | 1.00x | 100% |
| qft | 22 | one-thread-per-rank | 2 | 1 | 3.030 | 1.78x | 89% |
| qft | 22 | one-thread-per-rank | 4 | 1 | 2.360 | 2.28x | 57% |
| qft | 22 | one-thread-per-rank | 8 | 1 | 2.536 | 2.13x | 27% |

## Weak scaling (measured)

| circuit | thread policy | ranks | qubits | amplitudes/rank | wall (s) | efficiency |
|---|---|---:|---:|---:|---:|---:|
| ising | one-thread-per-rank | 1 | 20 | 1,048,576 | 0.417 | 100% |
| ising | one-thread-per-rank | 2 | 21 | 1,048,576 | 0.491 | 85% |
| ising | one-thread-per-rank | 4 | 22 | 1,048,576 | 0.730 | 57% |
| ising | one-thread-per-rank | 8 | 23 | 1,048,576 | 1.696 | 25% |

## Placement search quality (measured)

| qubits | ranks | candidate sets | samples | needing the heuristic | optimum found | worst gap | median speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 4 | 153 | 33 | 31 | 33/33 | 0.00% | 3x |
| 18 | 8 | 816 | 33 | 31 | 33/33 | 0.00% | 12x |
| 18 | 16 | 3,060 | 33 | 31 | 33/33 | 0.00% | 40x |

## Local kernel bandwidth (measured)

| kernel | threads | GB/s | in-place reference | fraction |
|---|---:|---:|---:|---:|
| cx | 1 | 18.3 | 31.6 | 58% |
| cx | 2 | 34.1 | 61.2 | 56% |
| cx | 4 | 40.9 | 73.4 | 56% |
| cx | 8 | 39.1 | 72.1 | 54% |
| cz | 1 | 35.4 | 31.6 | 112% |
| cz | 2 | 42.3 | 61.2 | 69% |
| cz | 4 | 41.8 | 73.4 | 57% |
| cz | 8 | 49.1 | 72.1 | 68% |
| h | 1 | 20.3 | 31.6 | 64% |
| h | 2 | 39.4 | 61.2 | 64% |
| h | 4 | 73.5 | 73.4 | 100% |
| h | 8 | 76.6 | 72.1 | 106% |
| rz | 1 | 32.1 | 31.6 | 101% |
| rz | 2 | 60.1 | 61.2 | 98% |
| rz | 4 | 79.2 | 73.4 | 108% |
| rz | 8 | 72.9 | 72.1 | 101% |
| swap | 1 | 39.6 | 31.6 | 125% |
| swap | 2 | 40.5 | 61.2 | 66% |
| swap | 4 | 41.2 | 73.4 | 56% |
| swap | 8 | 41.2 | 72.1 | 57% |

## Post-quantum primitives (measured)

| algorithm | operation | median (us) | bytes |
|---|---|---:|---:|
| ML-DSA-44 | keygen | 32.8 | 1312 |
| ML-DSA-44 | sign | 61.1 | 2420 |
| ML-DSA-44 | verify | 30.9 | 2420 |
| ML-DSA-65 | keygen | 55.0 | 1952 |
| ML-DSA-65 | sign | 95.7 | 3309 |
| ML-DSA-65 | verify | 49.5 | 3309 |
| ML-DSA-87 | keygen | 89.0 | 2592 |
| ML-DSA-87 | sign | 135.8 | 4627 |
| ML-DSA-87 | verify | 83.3 | 4627 |
| ML-KEM-1024 | decapsulate | 27.7 | 32 |
| ML-KEM-1024 | encapsulate | 24.2 | 1568 |
| ML-KEM-1024 | keygen | 23.5 | 1568 |
| ML-KEM-512 | decapsulate | 14.6 | 32 |
| ML-KEM-512 | encapsulate | 12.9 | 768 |
| ML-KEM-512 | keygen | 12.2 | 800 |
| ML-KEM-768 | decapsulate | 20.0 | 32 |
| ML-KEM-768 | encapsulate | 17.5 | 1088 |
| ML-KEM-768 | keygen | 17.1 | 1184 |

## Secure job envelope (measured)

| step | median (us) | bytes |
|---|---:|---:|
| bundle_size | - | 78059 |
| envelope_fixed_overhead | - | 7083 |
| pack_job | 3138 | 78059 |
| payload_base64 | - | 70976 |
| payload_plaintext | - | 53232 |
| verify_and_open | 5699 | 78059 |

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
