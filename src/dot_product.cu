#include <iostream>
#include <cassert> 
#include <cstdlib>
#include <ctime>

void check_dot_product(float *a, float *b, float *c, int N, const float tolerance) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float expected = 0.0f;
            for (int j = 0; j < N; j++) {
                expected += a[i * N + j] * b[j * N + k];
            }
            assert(std::fabs(expected - c[i * N + k]) < tolerance);
        }
    }
    std::cout << "Dot Product Calculation Successful" << "\n";
}

__global__ void dot_product(float *a, float *b, float *c, int N) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < N) {
        float temp_sum = 0.0f;
        for (int i = 0; i < N; i++) {
            temp_sum += a[row * N + i] * b[i * N + col];
        } 
        c[row * N + col] = temp_sum; 
    }
}

int main(int argc, char *argv[]) {

    int number = 1 << 8;
    srand(time(NULL));
    const float tolerance = 1e-4;
    int N = number;
    size_t size = N * N * sizeof(float);

    float *host_vector_a, *host_vector_b, *host_vector_c;
    float *device_vector_a, *device_vector_b, *device_vector_c;

    host_vector_a = (float*)malloc(size);
    host_vector_b = (float*)malloc(size);
    host_vector_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            host_vector_a[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
            host_vector_b[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    cudaMalloc(&device_vector_a, size);
    cudaMalloc(&device_vector_b, size);
    cudaMalloc(&device_vector_c, size);

    cudaMemcpy(device_vector_a, host_vector_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_b, host_vector_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_c, host_vector_c, size, cudaMemcpyHostToDevice);

    dim3 block_dimensions(8, 8);
    dim3 block_quantity(
        (N + block_dimensions.x - 1) / block_dimensions.x,
        (N + block_dimensions.y - 1) / block_dimensions.y
    );

    dot_product<<<block_quantity, block_dimensions>>>(device_vector_a, device_vector_b, device_vector_c, N);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Kernel Failed: " << cudaGetErrorString(error) << "\n";
    }

    cudaDeviceSynchronize();

    cudaMemcpy(host_vector_c, device_vector_c, size, cudaMemcpyDeviceToHost);

    check_dot_product(host_vector_a, host_vector_b, host_vector_c, N, tolerance);

    free(host_vector_a);
    free(host_vector_b);
    free(host_vector_c);

    cudaFree(device_vector_a);
    cudaFree(device_vector_b);
    cudaFree(device_vector_c);

    return 0;
}