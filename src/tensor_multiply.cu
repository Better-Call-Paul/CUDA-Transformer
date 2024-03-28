#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

#define SHMEM_SIZE 8 * 8 * 8 // shared memory size

void handle_cuda_error(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << "\n";
        exit(-1);
    }
}

/*
 * Checks kernel multiplation for 2-D Matrices
*/
void check_multiply_error_2d(float *a, float *b, float *c, int N, const float tolerance) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float expected = 0.0f;
            for (int k = 0; k < N; k++) {
                int a_idx = i * N + k, b_idx = k * N + j;
                expected += a[a_idx] * b[b_idx];
            }
            int c_idx = i * N + j;
            assert(std::fabs(c[c_idx] - expected) < tolerance);
        }
    }
    std::cout << "Kernel Multiplication Valid" << "\n";
}

/*
 * 2d Multiplication
*/
__global__ void tensor_multiply_2d(float *a, float *b, float *c, int N, int M) {
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

/*
 * Optimized 2D tensor multiplication
 * assumes square matrixes of dimensions N x N
*/
__global__ void cache_tiled_multiply_2d(float *a, float *b, float *c, int N, int tile_size) {

    __shared__ float A[SHMEM_SIZE];
    __shared__ float B[SHMEM_SIZE];

    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;

    float temp_sum = 0.0f;

    for (int i = 0; i < (N / tile_size); i++) {
        /*
         * Index Calculations
         * For A:
            row * n : global row (loop-invariant)
            i * tile_size : new set of colums each iteration
            thread.Idx : index of the column within the set
         * For B:
            i * tile_size * n : next set of rows each iteration
            threadIdx.y * n : row within that set
            column : index of the global column (loop-invariant)
        */
        A[(threadIdx.y * tile_size) + threadIdx.x] = a[row * N + (i * tile_size + threadIdx.x)];
        B[(threadIdx.y * tile_size) + threadIdx.x] = b[(i * tile_size * N + threadIdx.y * N) + col];

        // Make sure its loaded
        __syncthreads();

        for (int j = 0; j < tile_size; j++) {
            temp_sum += A[(threadIdx.y * tile_size) + j] * B[(j * tile_size) + threadIdx.x];
        }

        // Ensure no overwrites on shared memory values;
        __syncthreads();

    }
    c[(row * N) + col] = temp_sum;

}

int main(int argc, char *argv[]) {

    const float tolerance = 1e-4; // determines tolerance for all checks
    int number = 1 << 8;
    srand(time(NULL));
    float *host_vector_a, *host_vector_b, *host_vector_c;
    float *device_vector_a, *device_vector_b, *device_vector_c;

    
    // 2-D Matrix Calculations
    int N = number, M = number;
    int tile_size = 8; // for cache tiled operations
    size_t size = N * M * sizeof(float); 
    
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
    
    //tensor_multiply_2d<<<block_quantity, block_dimensions>>>(device_vector_a, device_vector_b, device_vector_c, N, M); // Naive
    cache_tiled_multiply_2d<<<block_quantity, block_dimensions>>>(device_vector_a, device_vector_b, device_vector_c, N, tile_size); // Cache Tiled


    cudaError_t error = cudaGetLastError();

    handle_cuda_error(error, "Kernel Failed");

    cudaDeviceSynchronize();

    cudaMemcpy(host_vector_c, device_vector_c, size, cudaMemcpyDeviceToHost);

    check_multiply_error_2d(host_vector_a, host_vector_b, host_vector_c, N, tolerance);

    //2-D tensor end 
    

    free(host_vector_a);
    free(host_vector_b);
    free(host_vector_c);

    cudaFree(device_vector_a);
    cudaFree(device_vector_b);
    cudaFree(device_vector_c);

    return 0;
}
