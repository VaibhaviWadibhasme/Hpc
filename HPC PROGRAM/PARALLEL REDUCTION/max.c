#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void parallel_reduction_max_sum(int* data, int size, int* max_val_ptr, int* sum_val_ptr) {
    // Initialize shared variables
    *max_val_ptr = data[0];
    *sum_val_ptr = 0;

    // Compute maximum and sum of each chunk in parallel
    #pragma omp parallel for reduction(max: *max_val_ptr) reduction(+: *sum_val_ptr)
    for (int i = 0; i < size; i++) {
        if (data[i] > *max_val_ptr) {
            *max_val_ptr = data[i];
        }
        *sum_val_ptr += data[i];
    }

    // Combine maximum and sum values from each chunk
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // Compute maximum value
            for (int i = 1; i < omp_get_num_threads(); i++) {
                int thread_max_val;
                #pragma omp critical
                {
                    thread_max_val = *max_val_ptr;
                }
                #pragma omp flush
                if (thread_max_val > *max_val_ptr) {
                    *max_val_ptr = thread_max_val;
                }
            }
        }
        #pragma omp section
        {
            // Compute sum value
            for (int i = 1; i < omp_get_num_threads(); i++) {
                int thread_sum_val;
                #pragma omp critical
                {
                    thread_sum_val = *sum_val_ptr;
                }
                #pragma omp flush
                *sum_val_ptr += thread_sum_val;
            }
        }
    }
}

int main() {
    int data_size = 1000000;
    int* data = malloc(data_size * sizeof(int));
    for (int i = 0; i < data_size; i++) {
        data[i] = rand() % 100;
    }
    int max_val, sum_val;
    parallel_reduction_max_sum(data, data_size, &max_val, &sum_val);
    printf("Maximum value: %d\n", max_val);
    printf("Sum value: %d\n", sum_val);
    free(data);
    return 0;
}
