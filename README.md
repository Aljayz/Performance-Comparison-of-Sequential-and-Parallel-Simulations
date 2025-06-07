# Parallel and Distributed Merge Sort Implementations

This repository provides multiple implementations of the Merge Sort algorithm with different parallelization approaches, including cluster-level distributed computing using MPI.

## Implementations

1. **Sequential Merge Sort** (seqMergesort.py)
   - Basic recursive implementation serving as performance baseline

2. **Shared-Memory Parallel Versions**:
   - Standard Python with multiprocessing (seqpar-mergesort.py)
   - NumPy-optimized version (numpy-seqpar-mergesort.py)
   - Supports 2-32 parallel processes on a single machine

3. **Distributed MPI Versions**:
   - Single-node MPI implementation (MPI-seq-par-mergesort.py)
   - Cluster-ready MPI implementation for multi-node execution

## Cluster MPI Execution

To run the MPI version across multiple nodes in a cluster:

```bash
mpiexec -hostfile hostfile -np <TOTAL_PROCESSES> python MPI-seq-par-mergesort.py
```

Where `hostfile` contains:
```
node1 slots=4
node2 slots=4
node3 slots=4
```

Key MPI features:
- Uses scatter-gather communication pattern
- Automatically balances workload across nodes
- Supports heterogeneous clusters
- Includes process synchronization

## Performance Characteristics

- Benchmarks from 50,000 to 2,000,000 elements
- Measures:
  - Sequential vs parallel execution times
  - Speedup and efficiency metrics
  - Strong and weak scaling
- Outputs CSV results for analysis

## Requirements

- Python 3.6+
- mpi4py package (for distributed versions)
- NumPy (for optimized shared-memory version)
- MPI runtime (OpenMPI or MPICH)

The implementations demonstrate parallel sorting across the full spectrum from single-machine multicore to distributed cluster environments.
