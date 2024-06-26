#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>
#include <cublas_v2.h>

/*
 * 3d matrix addition
 * 
*/
__global__ void mat_add_3d(float *a, float *b, float *c, int N, int M, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < N && column < M && depth < D) {
        int index = depth * (N * M) + (row * M) + column;
        c[index] = a[index] + b[index];
    }
}

/*
 * 2d matrix addition
*/
__global__ void mat_add_2d(float *a, float *b, float *c, int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < N && column < M) {
        int index = row * N + column;
        c[index] = a[index] + b[index];
    }
}

void check_addition_error_2d(float *a, float *b, float *c, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int idx = i * M + j;
            float expected = a[idx] + b[idx];
            assert(std::fabs(c[idx] - expected) < 1e-5);
        }
    }
}


void check_addition_error_3d(float* a, float* b, float* c, int N, int M, int D) {
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

/*
* For cublas 
*/
void cublas_vector_init(float *a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void cublas_verify(float *a, float *b, float *c, float factor, int N) {
    for (int i = 0; i < N; i++) {
        assert(c[i] == factor * a[i] + b[i]);
    }
    std::cout << "Cublas CUDA kernel addition verified successfully." << "\n";
}

int main(int argc, char *argv[]) {

    
    // 3D Vector Addition
    srand(time(NULL));
    int number = 1 << 8;
    int N = number, M = number, D = number;
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

    mat_add_3d<<<block_quantity, block_dimensions>>>(device_vector_a, device_vector_b, device_vector_c, N, M, D);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel Launch Failed: %s\n", cudaGetErrorString(error));
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(host_vector_c, device_vector_c, size, cudaMemcpyDeviceToHost);

    check_addition_error_3d(host_vector_a, host_vector_b, host_vector_c, N, M, D);


    free(host_vector_a);
    free(host_vector_b);
    free(host_vector_c);

    cudaFree(device_vector_a);
    cudaFree(device_vector_b);
    cudaFree(device_vector_c);

    // end of 3D vector addition
    

    /*
    // Cublas Version

    int N = 1 << 2;
    size_t bytes = N * sizeof(float);

    float *host_vector_a, *host_vector_b, *host_vector_c;
    float *device_vector_a, *device_vector_b;

    host_vector_a = (float*)malloc(bytes);
    host_vector_b = (float*)malloc(bytes);
    host_vector_c = (float*)malloc(bytes);

    cudaMalloc(&device_vector_a, bytes);
    cudaMalloc(&device_vector_b, bytes);

    cublas_vector_init(host_vector_a, N);
    cublas_vector_init(host_vector_b, N);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // copy host vectors to device instead of memcpy, last arg is step size
    cublasSetVector(N, sizeof(float), host_vector_a, 1, device_vector_a, 1); 
    cublasSetVector(N, sizeof(float), host_vector_b, 1, device_vector_b, 1); 

    const float scale = 2.0f;
    cublasSaxpy(handle, N, &scale, device_vector_a, 1, device_vector_b, 1);

    // copy result vector out
    cublasGetVector(N, sizeof(float), device_vector_b, 1, host_vector_c, 1);

    cublas_verify(host_vector_a, host_vector_b, host_vector_c, scale, N);

    cublasDestroy(handle);

    cudaFree(device_vector_a);
    cudaFree(device_vector_b);

    free(host_vector_a);
    free(host_vector_b);
    free(host_vector_c);

    // End of Cublas Version
    */

    return 0;
}