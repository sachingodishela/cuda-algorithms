#include<iostream>
#include<stdlib.h>
#include<cuda_runtime.h>

int main()
{
    std::cout << "Hello, CUDA!" << std::endl;
    int nDevices;
    cudaGetDeviceCount(&nDevices); 
    std::cout << "Number of CUDA devices: " << nDevices << std::endl;
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device Number: " << i << std::endl;
        std::cout << "Device Name: " << prop.name << std::endl;
        std::cout << "Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl; // Print global memory in MB
        std::cout << "Shared memory per MP: " << prop.sharedMemPerMultiprocessor / 1024 << " MB" << std::endl;
        std::cout << "Regs per multi processor: " << prop.regsPerMultiprocessor << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Max grid dimensions: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "Max shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
    return EXIT_SUCCESS;
}
