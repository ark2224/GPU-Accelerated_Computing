#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMultiply(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}


int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    // Matrix size in bytes
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = N * M * sizeof(float);

    // Declares device pointers
    float *dA, *dB, *dC;

    // Allocating device memory
    cudaMalloc(&dA, size_A);
    cudaMalloc(&dB, size_B);
    cudaMalloc(&dC, size_C);

    // Kernel Launch
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) /  blockDim.y);
    matrixMultiply<<<gridDim,blockDim>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    // Clean device's memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Check for device errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;
}
