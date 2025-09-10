#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    float *hA, *hB, *hC;
    float *dA, *dB, *dC;
    cudaStream_t stream1, stream2;

    // Allocate host memory
    hA = (float *)malloc(size);
    hB = (float *)malloc(size);
    hC = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < numElements; ++i) {
        hA[i] = rand() / (float)RAND_MAX;
        hB[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void **)&dA, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&dB, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&dC, size));

    // Create streams
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    // Copy inputs to device asynhronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dA, hA, size, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dB, hB, size, cudaMemcpyHostToDevice, stream2));

    // Launch kernels
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(dA, dB, dC, numElements);

    // copy result back to host asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(hC, dC, size, cudaMemcpyDeviceToHost, stream1));

    // Synchronize streams
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(hA[i] + hB[i] - hC[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test Passed\n");

    // Clean device and host memory
    CHECK_CUDA_ERROR(cudaFree(dA));
    CHECK_CUDA_ERROR(cudaFree(dB));
    CHECK_CUDA_ERROR(cudaFree(dC));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    free(hA);
    free(hB);
    free(hC);

    return 0;
}