#include<vector>
#include<cuda_runtime.h>

class Benchmark {
    Benchmark () {
        
    }
};

class Data {
public:
    float* gpu, *cpu;
    unsigned long long int s;

    Data(unsigned long long int size){
        this->s = size;
    }
};