#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000000; // Vector size
    int *a, *b, *c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors
    int size = n * sizeof(int); // Size in bytes

    // Allocate memory for host vectors
    a = (int*) malloc(size);
    b = (int*) malloc(size);
    c = (int*) malloc(size);

    // Initialize host vectors
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Allocate memory for device vectors
    cudaMalloc((void**) &d_a, size);
    cudaMalloc((void**) &d_b, size);
    cudaMalloc((void**) &d_c, size);

    // Copy host vectors to device vectors
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy device result vector to host result vector
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < n; i++) {
        if (c[i] != 2*i) {
            printf("Error: c[%d] = %d\n", i, c[i]);
            break;
        }
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}
