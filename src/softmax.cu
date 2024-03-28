#include <iostream>
#include <cassert>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <curand.h>


#define SHMEM_SIZE 8 * 8 * 8


__device__ void warp_reduce(volative float *shmeme_ptr, int thread_id) {

}

__global__ void sum_reduction(float *input, float *output) {

    __shared__ float partial_sum[SHMEM_SIZE];

    // calculate indexs

    // load elements and do first add reduction
    // vectir biw 2x as long as num of threads, scale i

    // store first partial result instead of just the elements
    for (int k = blockDix.x / 2; k > 32; k >>= 1) {

        if (threadIdx.x < k) {
            partial
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warp_reduce(partial_sum, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = partial_sum[0];
    }
}

__global__ void softmax(float* input, float *exponents, float *) {

}

int main(int argc, char *argv[]) {

    int number = 1 << 8;
    int N = number;
    size_t size = N * N * sizeof(float);
    const float epsilon = 1e-3;

    float *input_vector, *output_vector;
    float *exponent_holder, reduction_holder;
    float sum = 0.0f;
    
    cudaMalloc(input_vector, size);
    cudaMalloc(output_vector, size);
    cudaMalloc(exponent_holder, size);
    cudaMalloc(reduction_holder, size);


    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

    curandGenerateUniform(generator, srand(time(NULL)));

    











    




    return 0;
}