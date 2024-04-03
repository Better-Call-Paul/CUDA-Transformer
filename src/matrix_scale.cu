#include <iostream>
#include <curand.h>


__global__ void matrix_scale(float *input, float *output, float scale, int N, int M) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < N && col < M) {
        output[row * N + col] = input[row * N + col] * scale;
    }
}


int main() {

    int number = 1 << 8;
    int N = number, M = number;
    size_t size = N * M * sizeof(float);
    float scalar = 2.0f;

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


    dim3 block_dimension(8, 8);
    dim3 blocks_per_grid(
        (N + block_dimension.x - 1) / block_dimension.x,
        (M + block_dimension.y - 1) / block_dimension.y
    );

    matrix_scale<<<blocks_per_grid, block_dimension>>>(device_input_vector, device_output_vector, scalar, N, M);


    cudaMemcpy(host_output_vector, device_output_vector, size, cudaMemcpyDeviceToHost);


    cudaFree(device_input_vector); cudaFree(device_output_vector);
    free(host_input_vector); free(host_output_vector);

    return 0;
}