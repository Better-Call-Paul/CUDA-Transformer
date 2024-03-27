#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <cassert>

/*
 * Assumed to be in column major form
*/
void verify_solution(float *a, float *b, float *c, int n, const float epsilon) {
    float temp_sum;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp_sum = 0.0f;
            for (int k = 0; k < n; k++) {
                temp_sum += a[k * n + i] * b[j * n + k];
            }
            assert(std::fabs(c[j * n + i] - temp_sum) < epsilon);
        }
    }
    std::cout << "Cublas Kernel Successful" << "\n";
}

int main() {

    const float epsilon = 1e-3;
    int n = 1 << 10;
    size_t size = n * n * sizeof(float);


    float *host_vector_a, *host_vector_b, *host_vector_c;
    float *device_vector_a, *device_vector_b, *device_vector_c;

    host_vector_a = (float*)malloc(size);
    host_vector_b = (float*)malloc(size);
    host_vector_c = (float*)malloc(size);
    cudaMalloc(&device_vector_a, size);
    cudaMalloc(&device_vector_b, size);
    cudaMalloc(&device_vector_c, size);

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    // Fill matrixes on device
    curandGenerateUniform(prng, device_vector_a, n * n);
    curandGenerateUniform(prng, device_vector_b, n * n);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;


    // c = (alpha * a) * b + (beta * c)
    // (m x n) * (n x k) = (m x k)
    // Signature: handle, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc  : ld is leading dimension of x
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, device_vector_a, n, device_vector_b, n, &beta, device_vector_c, n); // CUBLAS_OP_N is reg matrix, OP_T is transpose

    cudaMemcpy(host_vector_a, device_vector_a, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vector_b, device_vector_b, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vector_c, device_vector_c, size, cudaMemcpyDeviceToHost);

    verify_solution(host_vector_a, host_vector_b, host_vector_c, n, epsilon);

    free(host_vector_a);
    free(host_vector_b);
    free(host_vector_c);

    cudaFree(device_vector_a);
    cudaFree(device_vector_b);
    cudaFree(device_vector_c);

    return 0;
}

