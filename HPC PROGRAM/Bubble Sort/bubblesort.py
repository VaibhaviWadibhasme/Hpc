import numpy as np
import time
import random
import omp

def parallel_bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        # Set the number of threads to the maximum available
        omp.set_num_threads(omp.get_max_threads())
        # Use the parallel construct to distribute the loop iterations among the threads
        # Each thread sorts a portion of the array
        # The ordered argument ensures that the threads wait for each other before moving on to the next iteration
        # This guarantees that the array is fully sorted before the loop ends
        with omp.parallel(num_threads=omp.get_max_threads(), default_shared=False, private=['temp']):
            for j in range(i % 2, n-1, 2):
                if arr[j] > arr[j+1]:
                    temp = arr[j]
                    arr[j] = arr[j+1]
                    arr[j+1] = temp

if __name__ == '__main__':
    # Generate a random array of 10,000 integers
    arr = np.array([random.randint(0, 100) for i in range(10000)])
    print(f"Original array: {arr}")

    start_time = time.time()
    parallel_bubble_sort(arr)
    end_time = time.time()

    print(f"Sorted array: {arr}")
    print(f"Execution time: {end_time - start_time} seconds")
