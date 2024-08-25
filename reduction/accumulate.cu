#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include "../benchmark/benchmark.h"
#define MAX_VECTOR_DIMENSIONS 134217728

Data _initialize_data(unsigned long long int &s)
{
    Data data(s);
    data.cpu = (float *)malloc(s * sizeof(float));
    data.cpu[0] = 0;
    for (unsigned long long int i = 1; i < s; i++)
    {
        data.cpu[i] = data.cpu[i - 1] + 0.00069;
    }
    size_t memorySize = s * sizeof(float);
    cudaError_t err = cudaMalloc(&data.gpu, memorySize);
    if (err != cudaSuccess)
    {
        std::string s = cudaGetErrorString(err);
        throw std::overflow_error(s.c_str());
        return 0;
    }
    cudaMemcpy(data.gpu, data.cpu, memorySize, cudaMemcpyHostToDevice);
    return data;
}

void _destroy_data(Data &d)
{
    cudaFree(d.gpu);
    free(d.cpu);
}

/**
 * Returns time consumed, in seconds
 */
float cpu_basic(Data data)
{
    auto tic = std::chrono::steady_clock::now();
    double solution = 0;
    for (int i = 0; i < data.s; i++)
    {
        solution += data.cpu[i];
    }
    auto toc = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
}

// kernel which executes on GPU parallely
__global__ void accumulate(float *data, unsigned long long int stride)
{
    int i = blockIdx.x * 1024 + threadIdx.x;
    if (i < stride)
    {
        data[i] = data[i] + data[stride + i];
    }
}

/**
 * Returns time consumed in seconds
 */
float gpu_basic(Data data)
{
    auto tic = std::chrono::steady_clock::now();
    // LOGIC
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int MAX_THREADS_PER_BLOCK = prop.maxThreadsPerBlock;
    for (unsigned long long int i = data.s / 2; i >= 1; i = i / 2)
    {
        int numBlocks = (i / MAX_THREADS_PER_BLOCK) + (i % MAX_THREADS_PER_BLOCK ? 1 : 0);
        int numThreadsPerBlock = numBlocks > 1 ? MAX_THREADS_PER_BLOCK : i;
        dim3 threadsPerBlock(numThreadsPerBlock);
        accumulate<<<numBlocks, threadsPerBlock>>>(data.gpu, i);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Error after calling the kernel: " << cudaGetErrorString(err);
            return EXIT_FAILURE;
        }
        cudaDeviceSynchronize();
    }
    float sum = 0;
    cudaMemcpy(&sum, data.gpu, sizeof(float), cudaMemcpyDeviceToHost);
    auto toc = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
}

int main()
{
    for (unsigned long long int s = 1; s <= 2e12; s *= 2)
    {
        Data data = _initialize_data(s);
        auto cpu_result = cpu_basic(data);
        auto gpu_result = gpu_basic(data);
        std::cout << "s: " << s << ", cpu_basic: " << cpu_result << ", gpu_basic: " << gpu_result << std::endl;
        _destroy_data(data);
    }
}