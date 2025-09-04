#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000 // Vector size of 10mil
#define BLOCK_SIZE 256

// Example:
// A = [1, 2, 3, 4, 5]
// B = [6, 7, 8, 9, 10]
// C = A + B = [7, 9, 11, 13, 15]

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    /*
        - a: float ptr of length n
        - b: vector of length n
        - c: sum vector of a and b; length n
    */
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU kernel for vector addition
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    // i := absolute thread index; imaging the matrix of threads being rolled flat
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if-statement for preventing incorrect memory access
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Measuring execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // Define CPU vectors
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    // Define GPU vectors
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // Allocating host memory for input vectors
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    // Initialize inputs
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);
    
    // Allocate device (GPU) memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy CPU data to GPU device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define GPU thread coordinates
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // initial test
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        // Synchronization adds barrier to coordinate disparate blocks
        cudaDeviceSynchronize();
    }

    // CPU Addition
    printf("Adding on CPU");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 100.0;

    // GPU Addition
    printf("Adding on CPU");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        double start_time = get_time();
        vector_add_cpu(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 100.0;

    // Results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("CPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup (cpu_time / gpu_time): %f\n", cpu_avg_time / gpu_avg_time);

    // Verification of results
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct within 1e-5" : "incorrect");

    // Clean memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    free(d_a);
    free(d_b);
    free(d_c);

    return 0;
}