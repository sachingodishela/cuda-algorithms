#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include<time.h>
#include<unistd.h>
#include<stdlib.h>
#include<vector>
#include<chrono>

__global__ void addVectors (float* A, float* B, unsigned long long int* vectorSize) {
    int i = blockIdx.x * 1024 + threadIdx.x;
    // if (threadIdx.x == 0) {
    //     printf("blockIdx.x: %d, threadIdx.x: %d, i: %d\n", blockIdx.x, threadIdx.x, i);
    // }
    if (i > *vectorSize) {
        // Do nothing
        return;
    }
    B[i] = A[i] + B[i];
}

std::vector<float*> createHostPointers (unsigned long long int &vectorSize) {
    float* A;
    float* B;
    float* C;
    A = (float*)malloc(vectorSize * sizeof(float));
    B = (float*)malloc(vectorSize * sizeof(float));
    for (unsigned long long int i = 0; i < vectorSize; i++) {
        A[i] = 1.0;
        B[i] = 2.2;
    }
    return std::vector<float*>{A, B};
}

void freeHostPointers (std::vector<float*> hostPointers) {
    for (int i = 0; i < hostPointers.size(); i++) {
        free(hostPointers[i]);
    }
}

std::vector<float*> createDevicePointers (std::vector<float*> &hostPointers, unsigned long long int &vectorSize) {
    cudaError_t err;
    size_t memorySize = vectorSize * sizeof(float);
    float* A = hostPointers[0];
    float* B = hostPointers[1];
    float* d_A;
    float* d_B;
    err = cudaMalloc(&d_A, memorySize);
    if (err != cudaSuccess) {
        std::cerr << "Error in allocating memory for A: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float*>{};
    }
    err = cudaMalloc(&d_B, memorySize);
    if (err != cudaSuccess) {
        std::cerr << "Error in allocating memory for B: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float*>{};
    }
    err = cudaMemcpy(d_A, A, memorySize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error in copying A: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float*>{};
    }
    err = cudaMemcpy(d_B, B, memorySize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error in copying B: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float*>{};
    }
    return std::vector<float*>{d_A, d_B};
}

void freeDevicePointers (std::vector<float*> devicePointers) {
    for (int i = 0; i < devicePointers.size(); i++) {
        cudaFree(devicePointers[i]);
    }
}

int main() {
    unsigned long long int vectorSize = 5;
    int iteration = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int MAX_THREADS_PER_BLOCK = prop.maxThreadsPerBlock;
    while (0 < vectorSize && vectorSize < 1e9) {
        std::cout << "vectorSize: " << vectorSize << ", "; 
        size_t memorySize = vectorSize * sizeof(float);
        std::cout << memorySize << " bytes, ";
        std::vector<float*> hostPointers = createHostPointers(vectorSize);
        auto tic = std::chrono::steady_clock::now();
        std::chrono::time_point<std::chrono::steady_clock> toc;
        int numBlocks = (vectorSize / MAX_THREADS_PER_BLOCK) + (vectorSize % MAX_THREADS_PER_BLOCK ? 1 : 0);
        std::cout << "iteration: " << ++iteration << ", ";
        // std::cout << "numBlocks: " << numBlocks << std::endl;
        int numThreadsPerBlock = numBlocks > 1 ? MAX_THREADS_PER_BLOCK : vectorSize;
        // std::cout << "numThreadsPerBlock: " << numThreadsPerBlock << std::endl;
        dim3 threadsPerBlock(numThreadsPerBlock);
        // create device pointers
        std::vector<float*> devicePointers = createDevicePointers(hostPointers, vectorSize);
        if (devicePointers.size() == 0) {
            return EXIT_SUCCESS;
        }
        unsigned long long int* d_vectorSize;
        cudaMalloc(&d_vectorSize, sizeof(unsigned long long int));
        cudaMemcpy(d_vectorSize, &vectorSize, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
        addVectors<<<numBlocks, threadsPerBlock>>>(devicePointers[0], devicePointers[1], d_vectorSize);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Error after calling the kernel: " << cudaGetErrorString(err);
            return EXIT_FAILURE;
        }
        cudaDeviceSynchronize();
        cudaMemcpy(hostPointers[1], devicePointers[1], memorySize, cudaMemcpyDeviceToHost);
        toc = std::chrono::steady_clock::now();
        freeDevicePointers(devicePointers);
        auto gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        // CPU
        auto tic2 = std::chrono::steady_clock::now();
        for (unsigned long long int i = 0; i < vectorSize; i++) {
            hostPointers[1][i] = hostPointers[0][i] + hostPointers[1][i];
        }
        freeHostPointers(hostPointers);
        auto toc2 = std::chrono::steady_clock::now();
        std::cout << "GPU/CPU time ratio: " << (float)gpuTime / std::chrono::duration_cast<std::chrono::milliseconds>(toc2 - tic2).count() << std::endl;
        
        vectorSize = vectorSize * 1.2;
    }
    return 0;
}