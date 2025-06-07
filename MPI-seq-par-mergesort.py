from mpi4py import MPI
import time
import csv
import numpy as np
from seqMergesort import merge, mergesort

def mergeSortParallelMPI(lyst):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    # Scatter the data
    if rank == 0:
        # Split the list into approximately equal parts
        data = np.array_split(lyst, size)
    else:
        data = None
    
    # Each process gets its portion of the data
    local_data = comm.scatter(data, root=0)
    
    # Local sorting
    local_sorted = mergesort(local_data.tolist() if isinstance(local_data, np.ndarray) else local_data)
    
    # Gather all sorted sublists at root
    gathered = comm.gather(local_sorted, root=0)
    
    # Root process merges all sorted sublists
    if rank == 0:
        while len(gathered) > 1:
            # Pair up adjacent sublists
            new_gathered = []
            for i in range(0, len(gathered), 2):
                if i + 1 < len(gathered):
                    merged = merge(gathered[i], gathered[i+1])
                    new_gathered.append(merged)
                else:
                    new_gathered.append(gathered[i])
            gathered = new_gathered
        return gathered[0]
    else:
        return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    sizes = [0, 50_000, 200_000, 400_000, 600_000, 800_000, 1_000_000, 1_500_000, 2_000_000]

    if rank == 0:
        results = [['N', 'Sequential', f'Parallel-{size}', 'Speed-up']]  # Added Speed-up column
    else:
        results = None
    
    for N in sizes:
        if rank == 0:
            lystbck = np.random.rand(N)
            row = [N]
            print(f"\nSize: {N}")

            # Sequential mergesort (only on rank 0)
            lyst = lystbck.copy()
            start = time.time()
            lyst = mergesort(lyst.tolist())
            seq_time = time.time() - start
            row.append(seq_time)
            print(f"Sequential: {seq_time:.4f} sec")
            
            # Parallel mergesort
            lyst = lystbck.copy()
            start = time.time()
            lyst = mergeSortParallelMPI(lyst.tolist())
            par_time = time.time() - start
            row.append(par_time)
            print(f"Parallel-{size}: {par_time:.4f} sec")
            
            # Calculate speed-up
            if seq_time > 0 and par_time > 0:
                speedup = seq_time / par_time
            else:
                speedup = 0.0
            row.append(speedup)
            print(f"Speed-up: {speedup:.2f}x")  # Display speed-up
            
            results.append(row)
        else:
            mergeSortParallelMPI(None)  # Other ranks participate but don't time
    
    if rank == 0:
        # Write results to CSV
        with open(f'../result/mpiMergeSortResults-Parallel-{size}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)

        print(f'\nResults saved to ../result/mpiMergeSortResults-Parallel-{size}.csv')

if __name__ == '__main__':
    main()