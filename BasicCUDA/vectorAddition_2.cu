#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 10000000 // Vector size 10mil
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for 1D vector addition (Naive)
__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x+ threadIdx.x;
    // one add, one multiple, one store PER thread
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for 3D vector addition (Naive)
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
    /*Inputs:
        - a: 3D matrix operand (nx, ny, nz)
        - b: 3D matrix operand (nx, ny, nz)
        - c: 3D matrix sum (nx, ny, nz)
        - nx: number of elements in x axis
        - ny: number of elements in y axis
        - nz: number of elements in z axis
    */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    // 3 addition ops, 3 multiple ops, 3 store ops
    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        if (idx < nx * ny * nz) {
            c[idx] = a[idx] + b[idx];
        }
    }
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Function to approximate execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // CPU matrices
    float *ha, *hb, *hc_cpu, *hc_gpu_1d, *hc_gpu_3d; 
    // GPU matrices
    float *da, *db, *dc_1d, *dc_3d;
    size_t size = sizeof(float);
    
    // Allocate host/CPU memory
    ha = (float*)malloc(size);
    hb = (float*)malloc(size);
    hc_cpu = (float*)malloc(size);
    hc_gpu_1d = (float*)malloc(size);
    hc_gpu_3d = (float*)malloc(size);

    // Initialize matrices
    srand(time(NULL));
    init_vector(ha, N);
    init_vector(hb, N);

    // Allocate device/GPU memory
    cudaMalloc(&da, size);
    cudaMalloc(&db, size);
    cudaMalloc(&dc_1d, size);
    cudaMalloc(&dc_3d, size);

    // Copy data from CPU to GPU
    cudaMemcpy(da, ha, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(db, hb, size, cudaMemcpyDeviceToHost);

    // define 1D grid and block dims
    int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    // define 3D grid and block dims
    int nx = 100, ny = 100, nz = 1000;
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z
    );

    // Quick warmup test
    for (int i = 0; i < 5; i ++) {
        vector_add_cpu(ha, hb, hc_cpu, N);
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(da, db, dc_1d, N);
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(da, db, dc_3d, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    // CPU Implementation
    double cpu_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        double start_time = get_time();
        vector_add_cpu(ha, hb, hc_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 100.0;

    // GPU 1D Implementation
    double gpu_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        // Clear previous result on GPU
        cudaMemset(dc_1d, 0, size);
        double start_time = get_time();
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(da, db, dc_1d, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_1d_avg_time = gpu_total_time / 100.0;

    // 1D results verification
    cudaMemcpy(hc_gpu_1d, dc_1d, size, cudaMemcpyDeviceToHost);
    bool correct_1d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(hc_cpu[i] - hc_gpu_1d[i]) > 1e-4) {
            correct_1d = false;
            std::cout << i << " cpu: " << hc_cpu[i] << " != " << hc_gpu_1d[i] << std::endl;
            break;
        }
    }
    printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

    // GPU 3D Implementation
    double gpu_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        cudaMemset(dc_3d, 0, size);
        double start_time = get_time();
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(da, db, dc_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_3d_avg_time = gpu_total_time / 100.0;

    // Verification of 3D
    cudaMemcpy(hc_gpu_3d, dc_3d, size, cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(hc_cpu[i] - hc_gpu_3d[i]) > 1e-5) {
            correct_3d = false;
            std::cout << i << " cpu: " << hc_cpu[i] << " != " << hc_gpu_3d[i] << std::endl;
            break;
        }
    }

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU 1D average time: %f milliseconds\n", gpu_1d_avg_time * 1000);
    printf("GPU 3D average time: %f milliseconds\n", gpu_3d_avg_time * 1000);
    printf("Speedup (CPU vs GPU 1D): %fx\n", cpu_avg_time / gpu_1d_avg_time);
    printf("Speedup (CPU vs GPU 3D): %fx\n", cpu_avg_time / gpu_3d_avg_time);
    printf("Speedup (GPU 1D vs GPU 3D): %fx\n", gpu_1d_avg_time / gpu_3d_avg_time);

    // Free memory
    free(ha);
    free(hb);
    free(hc_cpu);
    free(hc_gpu_1d);
    free(hc_gpu_3d);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc_1d);
    cudaFree(dc_3d);

    return 0;
}