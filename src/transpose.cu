#include <curand.h>
#include <iostream>


__global__ void transpose(float *input, float *output, int N, int M) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M && col < N) {
        output[row * M + col] = input[row * M + col];
    }
}

int main() {

    int number = 1 << 8;
    int N = number, M = number;
    size_t size = N * M * sizeof(float);

    float *host_input_vector, *host_output_vector, *device_input_vector, *device_output_vector;

    host_input_vector = (float*)malloc(size);
    host_output_vector = (float*)malloc(size);
    
    // Populate host vector
    curandGenerator_t generator;
    curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, (unsigned long long)clock());
    curandGenerateUniform(generator, host_input_vector, N * M);

    cudaMalloc(&device_input_vector, size);
    cudaMalloc(&device_output_vector, size);

    cudaMemcpy(device_input_vector, host_input_vector, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_output_vector, host_output_vector, size, cudaMemcpyHostToDevice);


    int threads_per_block = 256;
    int blocks_per_grid = (N * M + threads_per_block - 1) / threads_per_block;

    transpose<<<blocks_per_grid, threads_per_block>>>(device_input_vector, device_output_vector, N, M);


    cudaMemcpy(host_output_vector, device_output_vector, size, cudaMemcpyDeviceToHost);


    cudaFree(device_input_vector); cudaFree(device_output_vector);
    free(host_input_vector); free(host_output_vector);

    return 0;
}