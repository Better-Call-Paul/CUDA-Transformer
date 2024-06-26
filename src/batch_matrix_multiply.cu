#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cassert>


/*
 * Checks kernel multiplcation for 3-D matrices
 * P is the outer dimension of b
*/
void check_multiply_error_3d(float *a, float *b, float *c, int N, int M, int P, int D, const float tolerance) {
    for (int batch = 0; batch < D; batch++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < P; j++) {
                float expected = 0.0f;
                for (int k = 0; k < M; k++) {
                    int a_idx = batch * N * M + i * M + k;
                    int b_idx = batch * M * P + k * P + j;
                    expected += a[a_idx] * b[b_idx];
                }
                int c_idx = batch * N * P + i * P + j;
                assert(std::fabs(c[c_idx] - expected) < tolerance);
            }
        }
    }
    std::cout << "Kernel Multiplication Valid" << "\n";
}

/*
 * 3d Multiplication: Batched Matrix Multiplication
*/
__global__ void tensor_multiply_3d(float *a, float *b, float *c, int N, int M, int D) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int column = blockDim.y * blockIdx.y + threadIdx.y;
    int batch = blockDim.z * blockIdx.z + threadIdx.z;

    if (row < N && column < M && batch < D) {
        float temp_sum = 0.0f;
        for (int k = 0; k < N; k++) {
            temp_sum += a[batch * N * M + row * M + k] * b[batch * N * M + k * M + column];
        }
        c[batch * N * M + row * M + column] = temp_sum;
    }
}

void handle_cuda_error(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << "\n";
        exit(-1);
    }
}

int main(int argc, char *argv[]) {

    const float tolerance = 1e-4; // determines tolerance for all checks
    int number = 1 << 8;
    srand(time(NULL));
    float *host_vector_a, *host_vector_b, *host_vector_c;
    float *device_vector_a, *device_vector_b, *device_vector_c;

    
    // 3-D Tensor Calculations
    int N = number, M = number, D = number;
    size_t size = N * M * D * sizeof(float);

    host_vector_a = (float*)malloc(size);
    host_vector_b = (float*)malloc(size);
    host_vector_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < D; k++) {
                int index = k * (N * M) + j * N + i;
                host_vector_a[index] = static_cast<float>(rand()) / RAND_MAX;
                host_vector_b[index] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }

    cudaMalloc(&device_vector_a, size);
    cudaMalloc(&device_vector_b, size);
    cudaMalloc(&device_vector_c, size);

    cudaMemcpy(device_vector_a, host_vector_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_b, host_vector_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_c, host_vector_c, size, cudaMemcpyHostToDevice);

    dim3 block_dimensions(8, 8, 8);
    dim3 block_quantity(
        (N + block_dimensions.x - 1) / block_dimensions.x,
        (M + block_dimensions.y - 1) / block_dimensions.y,
        (D + block_dimensions.z - 1) / block_dimensions.z
    );

    tensor_multiply_3d<<<block_quantity, block_dimensions>>>(device_vector_a, device_vector_b, device_vector_c, N, M, D);

    cudaError_t error = cudaGetLastError();

    handle_cuda_error(error, "Kernel Failed");

    cudaDeviceSynchronize();

    cudaMemcpy(host_vector_c, device_vector_c, size, cudaMemcpyDeviceToHost);
    
    check_multiply_error_3d(host_vector_a, host_vector_b, host_vector_c, N, M, M, D, tolerance);
    // 3-D Tensor end
    
    free(host_vector_a);
    free(host_vector_b);
    free(host_vector_c);

    cudaFree(device_vector_a);
    cudaFree(device_vector_b);
    cudaFree(device_vector_c);

    return 0;
}