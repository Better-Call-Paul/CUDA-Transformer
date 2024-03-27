#include <iostream>
#include <ctime>
#include <cassert>
#include <cstdlib>

#define SHMEM_SIZE 8 * 8 * 8

void check_relu(float *c, float* default_c, int N, const float tolerance) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(default_c[i * N + j] == c[i * N + j]);
        }
    }
    std::cout << "Relu Valid" << "\n";
}

/*
 * Cache tiled Relu 
*/
__global__ void relu(float *c, int N, int tile_size) {

    __shared__ float C[SHMEM_SIZE];
    
    int row = blockDim.y  * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < N) {
        for (int i = 0; i < (N / tile_size); i++) {

            C[(threadIdx.y * tile_size) + threadIdx.x] = c[(row * N + (i * tile_size + threadIdx.x))];

            __syncthreads();

            for (int k = 0; k < tile_size; k++) {
                if (C[(threadIdx.y * tile_size) + k] < 0) {
                    C[(threadIdx.y * tile_size) + k] = 0;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {

    int number = 1 << 8, tile_size = 8;
    const float tolerance = 1e-4;
    int N = number;
    size_t size = N * N * sizeof(float);

    float *host_vector_c, *default_host_vector_c;
    float *device_vector_c;

    host_vector_c = (float*)malloc(size);
    default_host_vector_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            host_vector_c[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    default_host_vector_c = host_vector_c;

    cudaMalloc(&device_vector_c, size);

    cudaMemcpy(device_vector_c, host_vector_c, size, cudaMemcpyHostToDevice);

    dim3 block_dimensions(8, 8);
    dim3 block_quantity(
        (N + block_dimensions.x - 1) / block_dimensions.x,
        (N + block_dimensions.y - 1) / block_dimensions.y
    );

    relu<<<block_quantity, block_dimensions>>>(device_vector_c, N, tile_size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Kernel Failed: " << cudaGetErrorString(error) << "\n";
    }

    cudaDeviceSynchronize();

    cudaMemcpy(host_vector_c, device_vector_c, size, cudaMemcpyDeviceToHost);

    check_relu(host_vector_c, default_host_vector_c, N, tolerance);

    free(host_vector_c);

    cudaFree(device_vector_c);
    return 0;
}