#include <iostream>
#include <cuda_runtime.h>



__global__ void vector_add(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

__global__ void vector_multiply(float *A, float *B, float *C) 
{
    
}


int main() {
    int size = 100;
    float *host_vector_a = (float*)malloc(size * sizeof(float));
    float *host_vector_b = (float*)malloc(size * sizeof(float));
    float *host_vector_c = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        host_vector_a[i] = 1.0f;
        host_vector_b[i] = 1.0f;
        host_vector_c[i] = 1.0f;
    }

    float *device_vector_a;
    float *device_vector_b;
    float *device_vector_c;

    cudaMalloc(&device_vector_a, size * sizeof(float));
    cudaMalloc(&device_vector_b, size * sizeof(float));
    cudaMalloc(&device_vector_c, size * sizeof(float));

    cudaMemcpy(device_vector_a, host_vector_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_b, host_vector_b, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_c, host_vector_c, size * sizeof(float), cudaMemcpyHostToDevice);
    
    vector_add<<<1, size>>>(device_vector_a, device_vector_b, device_vector_c);


    cudaMemcpy(host_vector_a, device_vector_a, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vector_b, device_vector_b, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vector_c, device_vector_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) 
    {
        std::cout << "A: " << host_vector_a[i] << "\n";
        std::cout << "B: " << host_vector_b[i] << "\n";
        std::cout << "C: " << host_vector_c[i] << "\n";
    }
    free(host_vector_a);
    free(host_vector_b);
    free(host_vector_c);
    cudaFree(device_vector_a);
    cudaFree(device_vector_b);
    cudaFree(device_vector_c);

    std::cout << "Stable" << "\n";
    return 0;
}