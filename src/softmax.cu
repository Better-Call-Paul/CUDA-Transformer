#include <iostream>
#include <cassert>
#include <cmath>
#include <curand.h>
#include <cfloat>

#define SHMEM_SIZE 256


template<typename T>
__device__ T warpReduceMax(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template<typename T>
__device__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__global__ void softmax(T *input, T *output, int N, int M) {
    extern __shared__ T shared_data[];
    int tid = threadIdx.x;
    int row = blockIdx.x;

    T max_val = -FLT_MAX;
    T sum_val = 0;

    // Find max
    for(int i = tid; i < M; i += blockDim.x) {
        max_val = max(max_val, input[row * M + i]);
    }
    max_val = warpReduceMax(max_val);

    if (tid == 0) shared_data[blockIdx.x] = max_val;
    __syncthreads();
    max_val = shared_data[blockIdx.x];

    // Compute sum of exp(input - max)
    for(int i = tid; i < M; i += blockDim.x) {
        sum_val += exp(input[row * M + i] - max_val);
    }
    sum_val = warpReduceSum(sum_val);

    if(tid == 0) shared_data[blockIdx.x] = sum_val;
    __syncthreads();
    sum_val = shared_data[blockIdx.x];

    // Write the result
    for(int i = tid; i < M; i += blockDim.x) {
        output[row * M + i] = exp(input[row * M + i] - max_val) / sum_val;
    }
}

void handle_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
void check_softmax(float *input, float *output, float *computed_output, int N, int M, const float tolerance) {
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < M; j++) {
            computed_output[i * M + j] = exp(input[i * M + j]);
            sum += computed_output[i * M + j];
        }
        for (int k = 0; k < M; k++) {
            computed_output[i * M + k] /= sum;
            assert(std::fabs(computed_output[i * M + k] - output[i * M + k]) < tolerance);
        }
    }
}

int main() {
    const float tolerance = 1e-5;
    int number = 1 << 2;
    int N = number, M = number;
    size_t size = N * M * sizeof(float);

    float *host_input_vector, *host_output_vector, *device_input_vector, *device_output_vector, *iterative_output_vector;

    host_input_vector = (float*)malloc(size);
    host_output_vector = (float*)malloc(size);
    iterative_output_vector = (float*)malloc(size);
    
    // Populate host vector
    curandGenerator_t generator;
    curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, (unsigned long long)clock());
    curandGenerateUniform(generator, host_input_vector, N * M);

    cudaMalloc(&device_input_vector, size);
    cudaMalloc(&device_output_vector, size);

    cudaMemcpy(device_input_vector, host_input_vector, size, cudaMemcpyHostToDevice);


    int threads_per_block = 8;
    int blocks_per_grid = (N * M + threads_per_block - 1) / threads_per_block;

    softmax<<<blocks_per_grid, threads_per_block, SHMEM_SIZE * sizeof(float)>>>(device_input_vector, device_output_vector, N, M);
    
    handle_cuda_error(cudaGetLastError(), "Kernel Failed");

    cudaMemcpy(host_output_vector, device_output_vector, size, cudaMemcpyDeviceToHost);

    check_softmax(host_input_vector, host_output_vector, iterative_output_vector, N, M, tolerance);

    cudaFree(device_input_vector);
    cudaFree(device_output_vector);
    free(host_input_vector);
    free(host_output_vector);
    free(iterative_output_vector);
    curandDestroyGenerator(generator);

    return 0;
}