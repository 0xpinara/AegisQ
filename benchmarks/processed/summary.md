## Measurement host

- host: `pias-MacBook-Pro.local`
- CPU: Apple M2 (8 logical cores)
- OS: macOS-15.6.1-arm64-arm-64bit
- compiler: AppleClang 17.0.0.17000603
- MPI: Open MPI v5.0.10, package: Open MPI brew@Sequoia-arm64.local Distribution, ident: 5.0.10, repo rev: v5.0.10, Feb 23, 2026
- AegisQ 0.1.0 at commit `93b16fec05f0`

## Communication-aware placement (measured)

| circuit | qubits | ranks | baseline MPI bytes | optimized MPI bytes | reduction | baseline wall (s) | optimized wall (s) | wall change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ghz | 20 | 2 | 8,388,608 | 8,388,608 | 0.0% | 0.007 | 0.009 | +14.2% |
| ghz | 20 | 4 | 16,777,216 | 16,777,216 | 0.0% | 0.010 | 0.009 | -7.0% |
| ghz | 20 | 8 | 25,165,824 | 25,165,824 | 0.0% | 0.010 | 0.016 | +58.7% |
| grover | 20 | 2 | 536,870,912 | 150,994,944 | 71.9% | 0.465 | 0.452 | -2.8% |
| grover | 20 | 4 | 1,207,959,552 | 301,989,888 | 75.0% | 0.494 | 0.429 | -13.0% |
| grover | 20 | 8 | 1,879,048,192 | 520,093,696 | 72.3% | 0.566 | 0.444 | -21.6% |
| ising | 20 | 2 | 150,994,944 | 83,886,080 | 44.4% | 0.129 | 0.122 | -5.7% |
| ising | 20 | 4 | 301,989,888 | 234,881,024 | 22.2% | 0.138 | 0.126 | -9.0% |
| ising | 20 | 8 | 452,984,832 | 385,875,968 | 14.8% | 0.164 | 0.160 | -2.6% |
| qft | 20 | 2 | 343,932,928 | 25,165,824 | 92.7% | 0.349 | 0.316 | -9.3% |
| qft | 20 | 4 | 671,088,640 | 67,108,864 | 90.0% | 0.348 | 0.308 | -11.6% |
| qft | 20 | 8 | 981,467,136 | 125,829,120 | 87.2% | 0.358 | 0.283 | -20.9% |
| random | 20 | 2 | 159,383,552 | 83,886,080 | 47.4% | 0.140 | 0.100 | -28.5% |
| random | 20 | 4 | 251,658,240 | 167,772,160 | 33.3% | 0.113 | 0.100 | -11.5% |
| random | 20 | 8 | 411,041,792 | 251,658,240 | 38.8% | 0.166 | 0.129 | -22.1% |

## Strong scaling (measured)

| circuit | qubits | thread policy | ranks | threads/rank | wall (s) | speedup | efficiency |
|---|---:|---|---:|---:|---:|---:|---:|
| ising | 22 | one-thread-per-rank | 1 | 1 | 1.888 | 1.00x | 100% |
| ising | 22 | one-thread-per-rank | 2 | 1 | 1.023 | 1.85x | 92% |
| ising | 22 | one-thread-per-rank | 4 | 1 | 0.756 | 2.50x | 62% |
| ising | 22 | one-thread-per-rank | 8 | 1 | 0.798 | 2.36x | 30% |
| qft | 22 | one-thread-per-rank | 1 | 1 | 5.397 | 1.00x | 100% |
| qft | 22 | one-thread-per-rank | 2 | 1 | 3.029 | 1.78x | 89% |
| qft | 22 | one-thread-per-rank | 4 | 1 | 2.365 | 2.28x | 57% |
| qft | 22 | one-thread-per-rank | 8 | 1 | 2.417 | 2.23x | 28% |

## Weak scaling (measured)

| circuit | thread policy | ranks | qubits | amplitudes/rank | wall (s) | efficiency |
|---|---|---:|---:|---:|---:|---:|
| ising | one-thread-per-rank | 1 | 20 | 1,048,576 | 0.416 | 100% |
| ising | one-thread-per-rank | 2 | 21 | 1,048,576 | 0.482 | 86% |
| ising | one-thread-per-rank | 4 | 22 | 1,048,576 | 0.732 | 57% |
| ising | one-thread-per-rank | 8 | 23 | 1,048,576 | 1.750 | 24% |

## Placement search quality (measured)

| qubits | ranks | candidate sets | samples | needing the heuristic | optimum found | worst gap | median speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 4 | 153 | 33 | 31 | 33/33 | 0.00% | 3x |
| 18 | 8 | 816 | 33 | 31 | 33/33 | 0.00% | 12x |
| 18 | 16 | 3,060 | 33 | 31 | 33/33 | 0.00% | 40x |

## Local kernel bandwidth (measured)

| kernel | threads | GB/s | in-place reference | fraction |
|---|---:|---:|---:|---:|
| cx | 1 | 17.6 | 31.8 | 55% |
| cx | 2 | 35.3 | 60.7 | 58% |
| cx | 4 | 40.5 | 77.9 | 52% |
| cx | 8 | 40.6 | 72.0 | 56% |
| cz | 1 | 34.5 | 31.8 | 108% |
| cz | 2 | 40.8 | 60.7 | 67% |
| cz | 4 | 42.7 | 77.9 | 55% |
| cz | 8 | 48.4 | 72.0 | 67% |
| h | 1 | 20.2 | 31.8 | 63% |
| h | 2 | 39.5 | 60.7 | 65% |
| h | 4 | 71.5 | 77.9 | 92% |
| h | 8 | 76.5 | 72.0 | 106% |
| rz | 1 | 31.7 | 31.8 | 100% |
| rz | 2 | 61.6 | 60.7 | 101% |
| rz | 4 | 74.4 | 77.9 | 96% |
| rz | 8 | 71.9 | 72.0 | 100% |
| swap | 1 | 39.5 | 31.8 | 124% |
| swap | 2 | 39.9 | 60.7 | 66% |
| swap | 4 | 44.9 | 77.9 | 58% |
| swap | 8 | 42.2 | 72.0 | 59% |

## Post-quantum primitives (measured)

| algorithm | operation | median (us) | bytes |
|---|---|---:|---:|
| ML-DSA-44 | keygen | 32.8 | 1312 |
| ML-DSA-44 | sign | 62.2 | 2420 |
| ML-DSA-44 | verify | 30.9 | 2420 |
| ML-DSA-65 | keygen | 53.8 | 1952 |
| ML-DSA-65 | sign | 97.8 | 3309 |
| ML-DSA-65 | verify | 49.8 | 3309 |
| ML-DSA-87 | keygen | 89.3 | 2592 |
| ML-DSA-87 | sign | 137.0 | 4627 |
| ML-DSA-87 | verify | 83.3 | 4627 |
| ML-KEM-1024 | decapsulate | 27.8 | 32 |
| ML-KEM-1024 | encapsulate | 24.2 | 1568 |
| ML-KEM-1024 | keygen | 23.2 | 1568 |
| ML-KEM-512 | decapsulate | 14.6 | 32 |
| ML-KEM-512 | encapsulate | 13.0 | 768 |
| ML-KEM-512 | keygen | 12.2 | 800 |
| ML-KEM-768 | decapsulate | 20.0 | 32 |
| ML-KEM-768 | encapsulate | 17.5 | 1088 |
| ML-KEM-768 | keygen | 16.9 | 1184 |

## Secure job envelope (measured)

| step | median (us) | bytes |
|---|---:|---:|
| bundle_size | - | 78059 |
| envelope_fixed_overhead | - | 7083 |
| pack_job | 3119 | 78059 |
| payload_base64 | - | 70976 |
| payload_plaintext | - | 53232 |
| verify_and_open | 5783 | 78059 |

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
