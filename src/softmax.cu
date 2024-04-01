#include <iostream>
#include <cassert>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <curand.h>

#define SHMEM_SIZE 256 

__inline__ __device__ 
float warp_reduce(volatile float* shmem_ptr, int thread_id) {
    if (thread_id < 32) {
        shmem_ptr[thread_id] += (thread_id + 32 < blockDim.x) ? shmem_ptr[thread_id + 32] : 0;
        shmem_ptr[thread_id] += (thread_id + 16 < blockDim.x) ? shmem_ptr[thread_id + 16] : 0;
        shmem_ptr[thread_id] += (thread_id + 8 < blockDim.x) ? shmem_ptr[thread_id + 8] : 0;
        shmem_ptr[thread_id] += (thread_id + 4 < blockDim.x) ? shmem_ptr[thread_id + 4] : 0;
        shmem_ptr[thread_id] += (thread_id + 2 < blockDim.x) ? shmem_ptr[thread_id + 2] : 0;
        shmem_ptr[thread_id] += (thread_id + 1 < blockDim.x) ? shmem_ptr[thread_id + 1] : 0;
    }
    return shmem_ptr[thread_id];
}

__global__ void softmax(float *input, float *output, int N, int M) {
    
    extern __shared__ float shared_mem[];
    int thread_index = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Only compute exponentials for valid (non-padded) values 
    if (thread_index < N * M) {
        shared_mem[thread_index] = exp(shared_mem[thread_index]);
    }
    else {
        shared_mem[thread_index] = 0.0f;
    }

    // Ensure all exponentials are computed
    __syncthreads();

    // warp_reduce

    // get the value in the 0th index 
    if (threadIdx.x == 0) {
        
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

void handle_cuda_error(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << "\n";
        exit(-1);
    }
}

int main() {
    const float tolerance = 1e-3;
    int number = 1 << 8;
    int N = number, M = number;
    size_t size = N * M * sizeof(float);

    float *host_input_vector, *host_output_vector, *device_input_vector, *device_output_vector, *iterative_output_vector;

    host_input_vector = (float*)malloc(size);
    host_output_vector = (float*)malloc(size);
    iterative_output_vector = (float*)malloc(size);
    
    // Populate host vectors
    curandGenerator_t generator;
    curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, (unsigned long long)clock());
    curandGenerateUniform(generator, host_input_vector, N * M);

    cudaMalloc(&device_input_vector, size);
    cudaMalloc(&device_output_vector, size);

    cudaMemcpy(device_input_vector, host_input_vector, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_output_vector, host_output_vector, size, cudaMemcpyHostToDevice);


    dim3 block_dimensions(8, 8);

    dim3 threads(
        (N + block_dimensions.x - 1) / block_dimensions.x,
        (M + block_dimensions.y - 1) / block_dimensions.y
    );

    softmax<<<block_quantity, threads, SHMEM_SIZE * sizeof(float)>>>(device_input_vector, device_output_vector, N, M);
    
    handle_cuda_error(cudaGetLastError(), "Kernel Failed");

    cudaMemcpy(host_output_vector, device_output_vector, size, cudaMemcpyDeviceToHost);

    check_softmax(host_input_vector, host_output_vector, iterative_output_vector, N, M, tolerance);

    cudaFree(device_input_vector);
    cudaFree(device_output_vector);

    return 0;
}