#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

void handle_cuda_error(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

void check_multiply_error_2d(float *a, float *b, float *c, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float expected = 0.0f;
            for (int k = 0; k < N; k++) {
                int a_idx = i * N + k, b_idx = k * N + j;
                expected += a[a_idx] * b[b_idx];
            }
            int c_idx = i * N + j;
            assert(std::fabs(c[c_idx] - expected) < 1e-3);
        }
    }
}

void check_multiply_error_3d(float *a, float *b, float *c, int N, int M, int D) {

}

/*
 * 3d Multiplication
*/
__global__ void matrix_multiply_3d(float *a, float *b, float *c, int N, int M, int Z) {
    // so I get the indexes and multiply them together and add them into the matrix

}

/*
 * 2d Multiplication
*/
__global__ void matrix_multiply_2d(float *a, float *b, float *c, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    float temp_sum = 0.0f;

    if (row < N && column < M) {
        for (int i = 0; i < N; i++) {
            temp_sum += a[row * N + i] * b[N * i + column];
        }
        c[row * N + column] = temp_sum;
    }
}

int main(int argc, char *argv[]) {
    int number = 1 << 8;
    srand(time(NULL));
    int N = number, M = number;
    size_t size = N * M * sizeof(float);

    float *host_vector_a, *host_vector_b, *host_vector_c;
    host_vector_a = (float*)malloc(size);
    host_vector_b = (float*)malloc(size);
    host_vector_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {   
            int index = i * M + j;
            host_vector_a[index] = static_cast<float>(rand()) / RAND_MAX;
            host_vector_b[index] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float *device_vector_a, *device_vector_b, *device_vector_c;
    cudaMalloc(&device_vector_a, size);
    cudaMalloc(&device_vector_b, size);
    cudaMalloc(&device_vector_c, size);

    cudaMemcpy(device_vector_a, host_vector_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_b, host_vector_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_c, host_vector_c, size, cudaMemcpyHostToDevice);

    dim3 block_dimensions(8, 8);
    dim3 block_quantity(
        (N + block_dimensions.x - 1) / block_dimensions.x,
        (M + block_dimensions.y - 1) / block_dimensions.y
    );

    matrix_multiply_2d<<<block_quantity, block_dimensions>>>(device_vector_a, device_vector_b, device_vector_c, N, M);

    cudaError_t error = cudaGetLastError();

    handle_cuda_error(error, "Kernel Failed");

    cudaDeviceSynchronize();

    cudaMemcpy(host_vector_c, device_vector_c, size, cudaMemcpyDeviceToHost);

    check_multiply_error_2d(host_vector_a, host_vector_b, host_vector_c, N);

    free(host_vector_a);
    free(host_vector_b);
    free(host_vector_c);

    cudaFree(device_vector_a);
    cudaFree(device_vector_b);
    cudaFree(device_vector_c);

    return 0;
}