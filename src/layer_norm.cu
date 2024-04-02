#include <curand.h>
#include <iostream>
#include <cassert>
#include <cmath>

#define SHMEM_SIZE 256 

__inline__ __device__
float warp_reduce(float *shmem_ptr, int thread_id) {
    for (int i = 64; i > 0; i >>= 1) {
        shmem_ptr[thread_id] += (thread_id + i < threadIdx.x) ? shmem_ptr[thread_id + i] : 0;
    }
    return shmem_ptr[thread_id];
}

__global__ void mean() {

}

__global__ void variance() {

}


__global__ void layer_norm(float *d_input_vector, float *d_output_vector, int N, int M) {

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    



    

}

void check_layer_norm(float *h_input_vector, float *h_output_vector, float *iterative_output_vector, int N, int M, const float tolerance) {
    for (int i = 0; i < N; i++) {
        float sum = 0.0f, var = 0.0f, mean = 0.0f, var_sum = 0.0f;
        for (int j = 0; j < M; j++) {
            sum += h_input_vector[i * M + j];
        }
        mean = sum / M;
        for (int k = 0; k < M; k++) {
            var_sum += pow(abs(mean - h_input_vector[i * M + k]), 2.0);
        }
        var /= var_sum / M;
        for (int p = 0; p < M; p++) {
            iterative_output_vector[i * M + p] = abs(h_input_vector[i * M + p] / sqrtf(var + 1e-8));
            assert(std::fabs(h_input_vector[i * M + p] - iterative_output_vector[i * M + p]) - tolerance);
        }

    }
    std::cout << "Layer Norm Verified" << "\n";
}


int main() {

    int number = 1 << 8;
    int N = number, M = number;
    size_t size = N * M * sizeof(float);
    const float tolerance = 1e-4;

    float *host_input_vector, *host_output_vector, *iterative_output_vector, *device_input_vector, *device_output_vector;

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
    cudaMemcpy(device_output_vector, host_output_vector, size, cudaMemcpyHostToDevice);


    dim3 block_dimension(8, 8);
    dim3 blocks_per_grid(
        (N + block_dimension.x - 1) / block_dimension.x,
        (M + block_dimension.y - 1) / block_dimension.y
    );

    layer_norm<<<blocks_per_grid, block_dimension>>>(device_input_vector, device_output_vector, N, M);


    cudaMemcpy(host_output_vector, device_output_vector, size, cudaMemcpyDeviceToHost);

    check_layer_norm(host_input_vector, host_output_vector, iterative_output_vector, N, M, tolerance);


    cudaFree(device_input_vector); cudaFree(device_output_vector);
    free(host_input_vector); free(host_output_vector); free(iterative_output_vector);

    return 0;
}