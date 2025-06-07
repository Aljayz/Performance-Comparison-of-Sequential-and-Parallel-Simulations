from multiprocessing import Pool
import time, random, sys, csv
from seqMergesort import merge, mergesort

def mergeWrap(AandB):
    a, b = AandB
    return merge(a,b)

def mergeSortParallel(lyst,n=2):
    numproc = 2**n      #default 4 sublists to sort.
    endpoints = [int(x) for x in linspace(0, len(lyst), numproc+1)]
    args = [lyst[endpoints[i]:endpoints[i+1]]for i in range(numproc)]

    pool = Pool(processes=numproc)
    sortedsublists = pool.map(mergesort,args)

    while len(sortedsublists) > 1:
        #get sorted sublist pairs to send to merge
        args=[(sortedsublists[i], sortedsublists[i+1])
        for i in range(0,len(sortedsublists),2)]
        sortedsublists=pool.map(mergeWrap,args)
    return sortedsublists[0]

def linspace(a,b,nsteps):
    """
    return list of simple linear steps from a to b in nsteps.
    """
    ssize = float(b-a)/(nsteps-1)
    return[ a + i * ssize for i in range(nsteps)]

def main():
    sizes = [0, 
             50_000, 
             200_000, 
             400_000, 
             600_000, 
             800_000, 
             1_000_000, 
             1_500_000, 
             2000000]
    results = [['N', 
                'Sequential', 
                'Parallel-2', 
                'Parallel-4', 
                'Parallel-8', 
                'Parallel-16', 
                'Parallel-32']]
    
    for N in sizes:
        lystbck = [random.random() for _ in range(N)]
        row = [N]
        print(f"Size: {N}")
        # Sequential mergesort
        lyst = list(lystbck)
        start = time.time()
        lyst = mergesort(lyst)
        elapsed = time.time() - start
        row.append(elapsed)
        print(f"Sequential: {elapsed}")
        
        # Parallel mergesort with different number of processes(2^n)
        num = [1, 2, 3, 4, 5]
        for n in num:
            lyst = list(lystbck)
            start = time.time()
            lyst = mergeSortParallel(lyst, n)
            elapsed = time.time() - start
            row.append(elapsed)
            print(f"Parallel-{2**n}: {elapsed}")
        
        results.append(row)
    
    # Write results to CSV
    filepath = 'MergeSort/result/LocalMergeSort-Normal/SeqParMergeSort.csv'
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)
    
    print(f'Results saved to {filepath}')

if __name__ == '__main__':
    main()
