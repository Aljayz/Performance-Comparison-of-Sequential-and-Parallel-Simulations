# Performance Comparison of Sequential and Parallel Simulations
## Parallel Merge Sort Implementation  

This repository contains implementations of the Merge Sort algorithm using sequential, multiprocessing, and MPI-based parallel approaches.  

### Key Features  

- **Sequential Merge Sort** (`seqMergesort.py`)  
  Baseline implementation for performance comparison  

- **Multiprocessing Versions**  
  - Standard Python lists (`seqpar-mergesort.py`)  
  - NumPy-optimized (`numpy-seqpar-mergesort.py`)  
  - Supports 2-32 parallel processes  

- **MPI Version** (`MPI-seq-par-mergesort.py`)  
  - Distributed sorting across multiple nodes  
  - Uses scatter-gather pattern  

### How to Run  

1. Sequential/Multiprocessing:  
```bash
python seqpar-mergesort.py
python numpy-seqpar-mergesort.py
```

2. MPI Version (4 processes example):  
```bash
mpiexec -n 4 python MPI-seq-par-mergesort.py
```

### Performance Analysis  

- Benchmarks sorting from 50,000 to 2,000,000 elements  
- Measures execution time and speed-up  
- Outputs results to CSV files  

### Requirements  

- Python 3  
- mpi4py (for MPI version)  
- NumPy (for optimized version)  

The implementations demonstrate how parallel computing can significantly improve sorting performance for large datasets.
