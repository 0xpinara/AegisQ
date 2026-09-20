# benchmark methodology

> Status: placeholder. This document is written in the project phase that
> implements the corresponding component; see the phase checklist in the
> [README](../README.md).

## What "measured communication" means

Every byte reported by AegisQ is a byte the runtime handed to MPI. The
counters live in `CommunicationProfiler` and are incremented inside the same
function that issues the transfer, so a code path cannot move data without
being counted.

Counted:

- payload bytes of every `MPI_Sendrecv` issued by the runtime, in both
  directions,
- the number of pairwise exchanges, and which opcode caused each one,
- wall time spent inside those calls,
- collective calls (`MPI_Allreduce` for the norm, `MPI_Allgather` for the
  test-only gather).

Not counted:

- MPI's own protocol and envelope overhead,
- traffic MPI generates internally to implement a collective,
- anything that happens below the MPI interface (NIC, shared memory copies).

Reported byte counts are therefore a lower bound on wire traffic and an exact
account of what the algorithm asked for. That is the quantity the qubit
placement actually controls.

### Local versus world-reduced counters

`metrics()` reports one rank. `reduced_metrics()` reports the world:

- byte counts and call counts are **summed**, because the question is how much
  traffic the job generated in total;
- wall times are **maximised**, because a distributed run finishes when its
  slowest rank finishes.

### Verified byte accounting

The MPI test suite asserts closed-form byte counts for every placement — for
example, a single Hadamard on a global qubit moves exactly one full state
vector's worth of bytes summed over ranks (`2^n · 16` for fp64), and a `CX`
with a local control and a global target moves exactly half of that. These are
assertions, not documentation: an optimisation that changes the traffic
pattern has to change the expectations too.
