#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>

#define M 256
#define K 512
#define N 256
#define BLOCK_SIZE 32

// Example 3x2 @ 2x4 = 3x4 -> (M x K) @ (K x N) = (M x N)
// A = [[1, 2], 
//      [3, 4], 
//      [5, 6]]

// B = [[7, 8, 9, 10],
//      [11, 12, 13, 14]]

// C = A * B = [[1*7 + 2*11, 1*8 + 2*12, 1*9 + 2*13, 1*10 + 2*14],
//              [3*7 + 4*11, 3*8 + 4*12, 3*9 + 4*13, 3*10 + 4*14],
//              [5*7 + 6*11, 5*8 + 6*12, 5*9 + 6*13, 5*10 + 6*14]]

// C = [[29, 32, 35, 38],
//      [65, 72, 79, 86],
//      [101, 112, 123, 134]]

// CPU matmul
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// CUDA kernel implementation for matmul
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int l) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < l) {
        float sum = 0.0f;
        for (int j = 0; j < k; j++) {
            sum += A[row * k + j] * B[j * l + col];
        }
        C[row * l + col] = sum;
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Execution Time Measurement
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // CPU matrices and vectors
    float *hA, *hB, *hC_cpu, *hC_gpu;
    // GPU matrices and vectors
    float *dA, *dB, *dC;
    int size_A = M * K * sizeof(float);
    int size_B = N * K * sizeof(float);
    int size_C = M * N * sizeof(float);

    // CPU/host memory allocation
    hA = (float*)malloc(size_A);
    hB = (float*)malloc(size_B);
    hC_cpu = (float*)malloc(size_C);
    hC_gpu = (float*)malloc(size_C);

    // Random Init
    srand(time(NULL));
    init_matrix(hA, M, K);
    init_matrix(hB, K, N);

    // GPU/device memory allocation
    cudaMalloc(&dA, size_A);
    cudaMalloc(&dB, size_B);
    cudaMalloc(&dC, size_C);

    // Copy argument data to device
    cudaMemcpy(dA, hA, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size_B, cudaMemcpyHostToDevice);

    // Define device grid and block dimensions in 3D
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Initial Runs
    for (int i = 0; i < 3; i++) {
        matmul_cpu(hA, hB, hC_cpu, M, K, N);
        matmul_gpu(dA, dB, dC, M, K, N);
        cudaDeviceSynchronize();
    }

    // CPU matmul Run
    double cpu_total_time = 0.0;
    for (int i = 0; i < 50; i++) {
        double start_time = get_time();
        matmul_cpu(hA, hB, hC_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 50.0;

    // GPU matmul Run
    double gpu_total_time = 0.0;
    for (int i = 0; i < 50; i++) {
        cudaMemset(dC, 0, size_C);
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(dA, dB, dC, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 50.0;

    // Verification of matmul
    cudaMemcpy(hC_gpu, dC, size_C, cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(hC_cpu[i] - hC_gpu[i]) > 1e-5) {
            correct_3d = false;
            std::cout << i << " cpu: " << hC_cpu[i] << " != " << hC_gpu[i] << std::endl;
            break;
        }
    }

    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Free memory
    free(hA);
    free(hB);
    free(hC_cpu);
    free(hC_gpu);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}