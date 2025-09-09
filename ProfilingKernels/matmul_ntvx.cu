#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void matrixMulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMul(float *A, float *B, float *C, int N) {
    nvtxRangePush(L"Matrix Multiplication");

    float *dA, *dB, *dC;
    int size = N * N * sizeof(float);

    nvtxRangePush(L"Memory Allocation"); // Start malloc profiling
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);
    nvtxRangePop(); // End malloc profiling

    nvtxRangePush(L"Memory Copy H2D"); // Start H2D Copy profiling
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
    nvtxRangePop(); // End H2D Copy profiling

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    nvtxRangePush(L"Kernel Execution"); // Start Kernel Ex profiling
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();
    nvtxRangePop(); // End Kernel Ex profiling

    nvtxRangePush(L"Memory Copy D2H"); // Start D2H Copy profiling
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);
    nvtxRangePop(); // End D2H Copy profiling
    
    nvtxRangePush(L"Memory Deallocation"); // Start Memory dealloc profiling
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    nvtxRangePop(); // End Memory dealloc profiling

    nvtxRangePop(); // End Matrix Mul
}


int main() {
    const int N = 1024;
    float *A = new float[N*N];
    float *B = new float[N*N];
    float *C = new float[N*N];

    matrixMul(A, B, C, N);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}