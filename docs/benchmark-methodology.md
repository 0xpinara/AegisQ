# Benchmark methodology

How the numbers in the README and the technical report were produced, and
which of them mean anything.

Everything here is measured on one machine. Nothing is extrapolated, modelled
or copied from another paper. Where a quantity cannot be measured well enough
to support a claim, the claim is not made — that is what most of this document
is about.

## Two kinds of number

The distinction matters more than any other in this repository.

| | Bytes and messages | Wall time |
|---|---|---|
| Source | counters in the runtime, incremented where a buffer is handed to MPI | `time.perf_counter()` around the run |
| Varies between identical runs | never | by tens of percent, measured below |
| Determined by | the circuit and the placement | the circuit, the placement, and the machine's mood |
| Used for | the placement result | context only, and only when resolved |

Communication volume is a deterministic function of the circuit and the qubit
placement, and the distributed test suite asserts the exact byte counts. When
the README says placement removed 87.2% of the traffic for a QFT, that is a
count, not an estimate.

Wall time is not like that, and the rest of this document is about how far it
can be trusted.

## What a single measurement is

One *launch* is one process (or one `mpirun` world of processes) that:

1. builds the circuit,
2. runs it once as a **warm-up** and discards the result — the first touch of
   a fresh shard pays page-fault and allocation costs that say nothing about
   the steady-state cost of the circuit,
3. runs it `--repeats` more times (default 3), timing each,
4. appends one raw row per timed repeat.

A launch is reduced to a number by taking the **minimum** wall time over its
repeats. Section [Why the minimum](#why-the-minimum) explains why.

## Thread policy is an explicit variable

Comparing rank counts means deciding what happens to the cores, and the two
reasonable answers measure different things. Neither is combined with the
other in a single table.

| Policy | Threads per rank | What it measures |
|---|---|---|
| `one-thread-per-rank` | 1 | classical strong scaling: total cores grow with the rank count |
| `fixed-total-cores` | `cores / ranks` | the cost of partitioning, with hardware held constant |

Under `fixed-total-cores` a speedup above 1 would mean partitioning itself
helped; on a single socket it generally does not, and the measurements say so.

## Calibrating the harness

Before quoting any wall-time difference, the setup is asked how small a
difference it can see. The answer is measured, not assumed.

For each configuration, **the identical run is launched twice** and the
apparent change `(t₂ − t₁) / t₁` recorded, ten times over. Nothing differs
between the two launches, so every value is zero by construction; whatever the
clock reports is the instrument. This is an A/A test.

```
aegisq benchmark calibrate --circuits ghz,qft,ising,grover,random \
    --qubits 20 --ranks 2,4,8 --trials 10
```

It reports a great deal of scatter. The largest apparent change with no change
made is in the tens of percent, and even the quietest configuration — the
longest-running circuit in the set — moves by several percent. The current
numbers are in the README under *What this harness can actually resolve* and
in `benchmarks/processed/calibration.csv`.

This is not a formality. An earlier version of the placement table reported a
wall-time change of −30.8% for a GHZ circuit whose byte count the optimiser
had left *exactly* unchanged. The true effect there is zero by construction,
and the A/A test shows the harness producing changes of that size on its own.

### Resolution, and the claim it supports

The **resolution** of a configuration is the largest apparent change over its
null trials. The definition is chosen so that the statement it licenses is
exact:

> An observed effect larger than every one of `n` null trials has probability
> at most `1 / (n + 1)` of arising by chance, if the two conditions were
> exchangeable.

With ten trials that is `p ≤ 0.09`. It is a weak bound, and it is stated
rather than dressed up. Note that it requires comparing against a value the
trials actually produced — an interpolated quantile lands below the largest
observation and silently weakens the bound.

A wall-time change is reported as **resolved** only if it

1. exceeds the resolution measured for *that* configuration, and
2. has a repeat range that does not overlap the baseline's.

The resolution is measured per circuit family, not once for the machine, because
a GHZ chain finishing in eight milliseconds and a Grover circuit running for
half a second are not equally timeable. A single machine-wide floor would
either excuse the first's noise or discard the second's real effects.

### Why the minimum

Interference on this machine is **one-sided**. Nothing schedules a launch to
run faster than an uncontended one; scheduling, thermal behaviour and
background load only ever make a run slower. The measured distribution of
launch times is right-skewed with a long upper tail, which is exactly that
shape.

Two consequences:

- The minimum is the right location estimator. It converges on the uncontended
  runtime as samples are added, while a mean chases the tail.
- The minimum should be taken **across launches**, not only across repeats
  inside one launch. Repeats within a launch share that launch's offset, so
  averaging them does not remove the dominant noise source.

Resampling the measured null puts the median resolution at roughly 27% for one
launch per arm, 10% for three and 6% for five. The placement sweep therefore
runs each configuration several times:

```
aegisq benchmark mapping --launches 5 ...
```

with the launch loop *outside* the treatments, so each round visits every
treatment once. Raw rows carry a `launch` column, and the analysis projects the
resolution for whatever number of launches a figure was actually reduced from.

### An order effect that is not there

The sweep runs `default` before `optimized`, which would bias the comparison if
machine state drifted over the seconds between them. An A/A test that compares
the first launch against the second, across many trials, finds no systematic
sign: the differences scatter around zero. The round-robin ordering above is
kept because it costs nothing, not because it corrected anything.

## Provenance

Every raw row records the host, CPU, OS, toolchain, AegisQ version, the git
commit, and whether the source tree had uncommitted changes when the
measurement ran. The report names every commit present in the data and warns
if any row came from a modified tree, so a figure can always be traced to the
code that produced it.

Raw measurements live in `benchmarks/raw/` and are never edited. Everything in
`benchmarks/processed/`, every plot, every LaTeX table and the README results
block is regenerated from them by:

```
python scripts/generate_report.py
```

`--check` fails if the committed README no longer follows from the committed
raw data; CI runs it on every push.

## Reproducing the whole thing

```
./scripts/benchmark_local.sh
```

runs the calibration, the optimisation levers, strong and weak scaling, kernel
bandwidth, placement search quality, single-precision error, the post-quantum
primitives and the Grover query scaling, then regenerates every derived
artefact. Sizes are configurable:

```
QUBITS=22 RANKS=2,4 LAUNCHES=3 ./scripts/benchmark_local.sh
```

The results describe the host they ran on. They are not a claim about cluster
hardware, and `docs/limitations.md` says what else they are not.
