#include <iostream>
#include <cassert>
#include <ctime>
#include <cstdlib>
#include <cmath>


__global__ void softmax(float* input, float *exponents, float *) {

}

int main(int argc, char *argv[]) {

    int number = 1 << 8;
    int N = number;
    size_t size = N * N * sizeof(float);

    float *input_vector, *output_vector;
    float *exponent_holder, reduction_holder;
    float sum = 0.0f;
    
    cudaMalloc(input_vector, size);
    cudaMalloc(output_vector, size);
    cudaMalloc(exponent_holder, size);
    cudaMalloc(reduction_holder, size);













    




    return 0;
}