#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

__global__ void mat_add(float *a, float *b, float *c, int N, int M, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < N && column < M && depth < D) {
        int index = depth * (N * M) + (row * M) + column;
        c[index] = a[index] + b[index];
    }
}


void check_addition(float* a, float* b, float* c, int N, int M, int D) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < D; k++) {
                int idx = k * (N * M) + (i * M) + j;
                float expected = a[idx] + b[idx];
                assert(std::fabs(c[idx] - expected) < 1e-5);
            }
        }
    }
    std::cout << "CUDA kernel addition verified successfully." << "\n";
}

int main(int argc, char *argv[]) {

    srand(time(NULL));

    int N = 100, M = 100, D = 100;
    size_t size = N * M * D * sizeof(float);
    float *host_vector_a, *host_vector_b, *host_vector_c;

    host_vector_a = (float*)malloc(size);
    host_vector_b = (float*)malloc(size);
    host_vector_c = (float*)malloc(size);


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < D; k++) {
                int idx = k * (N * M) + (i * M) + j;
                host_vector_a[idx] = static_cast<float>(rand()) / RAND_MAX;
                host_vector_b[idx] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }

    float *device_vector_a, *device_vector_b, *device_vector_c;

    cudaMalloc(&device_vector_a, size);
    cudaMalloc(&device_vector_b, size);
    cudaMalloc(&device_vector_c, size);

    cudaMemcpy(device_vector_a, host_vector_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_b, host_vector_b, size, cudaMemcpyHostToDevice);

    dim3 block_dimensions(8, 8, 8);
    dim3 block_quantity(
        (block_dimensions.x + N - 1) / block_dimensions.x,
        (block_dimensions.y + M - 1) / block_dimensions.y,
        (block_dimensions.z + D - 1) / block_dimensions.z
    );

    mat_add<<<block_quantity, block_dimensions>>>(device_vector_a, device_vector_b, device_vector_c, N, M, D);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel Launch Failed: %s\n", cudaGetErrorString(error));
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(host_vector_c, device_vector_c, size, cudaMemcpyDeviceToHost);

    check_addition(host_vector_a, host_vector_b, host_vector_c, N, M, D);
    

    free(host_vector_a);
    free(host_vector_b);
    free(host_vector_c);

    cudaFree(device_vector_a);
    cudaFree(device_vector_b);
    cudaFree(device_vector_c);

    return 0;
}