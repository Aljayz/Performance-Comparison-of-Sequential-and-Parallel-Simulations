from multiprocessing import Pool
import time
import csv
import numpy as np
from seqMergesort import merge, mergesort

def mergeWrap(AandB):
    a, b = AandB
    return merge(a, b)

def mergeSortParallel(lyst, n=2):
    numproc = 2**n  # Default 4 sublists to sort.
    endpoints = np.linspace(0, len(lyst), numproc + 1, dtype=int)
    args = [lyst[endpoints[i]:endpoints[i + 1]] for i in range(numproc)]

    pool = Pool(processes=numproc)
    sortedsublists = pool.map(mergesort, args)

    while len(sortedsublists) > 1:
        # Get sorted sublist pairs to send to merge
        args = [(sortedsublists[i], sortedsublists[i + 1]) for i in range(0, len(sortedsublists), 2)]
        sortedsublists = pool.map(mergeWrap, args)
    
    return sortedsublists[0]

def main():
    sizes = [0, 50_000, 200_000, 400_000, 600_000, 800_000, 1_000_000, 1_500_000, 2_000_000]
    results = [['N', 'Sequential', 'Parallel-2', 'Parallel-4', 'Parallel-8', 'Parallel-16', 'Parallel-32']]
    base_output_dir = "D:/Albert/SUBJECTS/CSC175/Projects/MergeSort/result/LocalMergeSort-Numpy/"
    
    
    for N in sizes:
        lystbck = np.random.rand(N)  # Numpy for random number generation
        row = [N]
        print(f"Size: {N}")

        # Sequential mergesort
        lyst = lystbck.copy()
        start = time.time()
        lyst = mergesort(lyst.tolist())  # Convert back to list for compatibility
        elapsed = time.time() - start
        row.append(elapsed)
        print(f"Sequential: {elapsed:.4f} sec")
        
        # Parallel mergesort with different number of processes(2^n)
        num = [1, 2, 3, 4, 5]
        for n in num:
            lyst = lystbck.copy()
            start = time.time()
            lyst = mergeSortParallel(lyst.tolist(), n)
            elapsed = time.time() - start
            row.append(elapsed)
            print(f"Parallel-{2**n}: {elapsed:.4f} sec")
        
        results.append(row)
    
    # Write results to CSV
    output_file = base_output_dir + 'npSeqParMergeSort.csv'
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    print(f'Results saved to {output_file}')

if __name__ == '__main__':
    main()
