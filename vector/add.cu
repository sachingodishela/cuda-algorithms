#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include<time.h>
#include<unistd.h>
#include<stdlib.h>
#include<vector>
#include<chrono>

__global__ void addVectors (float* A, float* B, float* C, unsigned long long int* vectorSize) {
    int i = blockIdx.x * 1024 + threadIdx.x;
    if (i > *vectorSize) {
        // Do nothing
        return;
    }
    C[i] = A[i] + B[i];
}

std::vector<float*> createDevicePointers (std::vector<float> A, std::vector<float> B, std::vector<float> C) {
    cudaError_t err;
    size_t memorySize = A.size() * sizeof(float);
    float* d_A;
    float* d_B;
    float* d_C;
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
    err = cudaMalloc(&d_C, memorySize);
    if (err != cudaSuccess) {
        std::cerr << "Error in allocating memory for C: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float*>{};
    }
    err = cudaMemcpy(d_A, A.data(), memorySize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error in copying A: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float*>{};
    }
    err = cudaMemcpy(d_B, B.data(), memorySize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error in copying B: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float*>{};
    }
    err = cudaMemcpy(d_C, C.data(), memorySize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error in copying C: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float*>{};
    }
    return std::vector<float*>{d_A, d_B, d_C};
}

void freeDevicePointers (std::vector<float*> devicePointers) {
    for (int i = 0; i < devicePointers.size(); i++) {
        cudaFree(devicePointers[i]);
    }
}

int main() {
    unsigned long long int vectorSize = 1;
    int iteration = 0;
    int MAX_THREADS_PER_BLOCK = 1024;
    int MAX_BLOCKS = 32;
    while (0 < vectorSize && vectorSize < ULONG_LONG_MAX) {
        std::cout << "vectorSize: " << vectorSize << ", "; 
        size_t memorySize = vectorSize * sizeof(float);
        std::vector<float> A(vectorSize);
        std::vector<float> B(vectorSize);
        std::vector<float> C(vectorSize, 0);
        for (int i = 0; i < vectorSize; i++) {
            A[i] = i+1;
            B[i] = i+1;
        }
        auto tic = std::chrono::steady_clock::now();
        int numBlocks = (vectorSize / MAX_THREADS_PER_BLOCK) + (vectorSize % MAX_THREADS_PER_BLOCK ? 1 : 0);
        std::cout << "iteration: " << ++iteration << ", ";
        if (numBlocks < MAX_BLOCKS) {
            // std::cout << "numBlocks: " << numBlocks << std::endl;
            int numThreadsPerBlock = numBlocks > 1 ? MAX_THREADS_PER_BLOCK : vectorSize;
            // std::cout << "numThreadsPerBlock: " << numThreadsPerBlock << std::endl;
            dim3 threadsPerBlock(numThreadsPerBlock);
            // create device pointers
            std::vector<float*> devicePointers = createDevicePointers(A, B, C);
            if (devicePointers.size() == 0) {
                std::cerr << "Error: unable to create device pointers" << std::endl;
                return EXIT_FAILURE;
            }
            unsigned long long int* d_vectorSize;
            cudaMalloc(&d_vectorSize, sizeof(unsigned long long int));
            cudaMemcpy(d_vectorSize, &vectorSize, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
            addVectors<<<numBlocks, threadsPerBlock>>>(devicePointers[0], devicePointers[1], devicePointers[2], d_vectorSize);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Error after calling the kernel: " << cudaGetErrorString(err);
                return EXIT_FAILURE;
            }
            cudaDeviceSynchronize();
            cudaMemcpy(C.data(), devicePointers[2], memorySize, cudaMemcpyDeviceToHost);
            auto toc = std::chrono::steady_clock::now();
            freeDevicePointers(devicePointers);
            std::cout <<  "num threads: " << numThreadsPerBlock << ", numBlocks: " << numBlocks << ", time taken using GPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count() << " ms, ";
        } else {
            // skip GPU execution
        }

        // CPU
        tic = std::chrono::steady_clock::now();
        for (unsigned long long int i = 0; i < vectorSize; i++) {
            C[i] = A[i] + B[i];
        }
        auto toc2 = std::chrono::steady_clock::now();
        std::cout << "time taken using CPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(toc2 - tic).count() << " ms" << std::endl;
        vectorSize = vectorSize * 2;
    }
    return 0;
}