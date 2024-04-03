#include <curand.h>
#include <iostream>
#include <cassert>
#include <cmath>


__global__ void sum_values(float *d_input, float *d_mean, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0f;

    if (i < N) {
        sum = d_input[i];
    }

    sdata[tid] = sum;

    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i]; 
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_mean, sdata[0]);
    }

}

__global__ void sum_variance(float *d_input, float *d_var, float *d_mean, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;
    if (i < N) {
        float diff = *d_mean - d_input[i];
        sum += diff * diff;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_var, sdata[0]);
    }
}


__global__ void layer_norm(float *d_input_vector, float *d_output_vector, float *d_mean, float *d_var, int N) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        d_output_vector[i] = (d_input_vector[i] - *d_mean) / (sqrtf(*d_var + 1e-8));
    }
}

void check_layer_norm(float *h_input_vector, float *h_output_vector, float *iterative_output_vector, int N, const float tolerance) {
    float sum = 0.0f, var = 0.0f, mean = 0.0f, var_sum = 0.0f;
    
    for (int i = 0; i < N; i++) {
        sum += h_input_vector[i];
    }
    mean = sum / N;

    for (int i = 0; i < N; i++) {
        var_sum += pow((mean - h_input_vector[i]), 2.0);
    }
    var = var_sum / N;
    
    for (int i = 0; i < N; i++) {
        iterative_output_vector[i] = (h_input_vector[i] - mean) / sqrtf(var + 1e-8);
        assert(std::fabs(h_output_vector[i] - iterative_output_vector[i]) < tolerance);
    }

    std::cout << "Layer Norm Verified" << "\n";
}



int main() {

    int N = 1 << 8;
    size_t size = N * sizeof(float);
    const float tolerance = 1e-5;

    float *host_input_vector, *host_output_vector, *iterative_output_vector, *device_input_vector, *device_output_vector;
    float *mean, *variance, *device_mean, *device_variance;

    mean = (float*)malloc(sizeof(float));
    variance = (float*)malloc(sizeof(float));
    host_input_vector = (float*)malloc(size);
    host_output_vector = (float*)malloc(size);
    iterative_output_vector = (float*)malloc(size);

    *variance = 0.0f; *mean = 0.0f;
    
    // Populate host vector
    curandGenerator_t generator;
    curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, (unsigned long long)clock());
    curandGenerateUniform(generator, host_input_vector, N);

    cudaMalloc(&device_input_vector, size);
    cudaMalloc(&device_output_vector, size);
    cudaMalloc(&device_mean, sizeof(float));
    cudaMalloc(&device_variance, sizeof(float));

    cudaMemcpy(device_input_vector, host_input_vector, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_output_vector, host_output_vector, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mean, mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_variance, variance, sizeof(float), cudaMemcpyHostToDevice);


    int threads_per_block = 8;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Can call mean, var from kernel with dynamic parallelism

    sum_values<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(device_input_vector, device_mean, N);

    cudaMemcpy(mean, device_mean, sizeof(float), cudaMemcpyDeviceToHost);

    *mean /= N;

    cudaMemcpy(device_mean, mean, sizeof(float), cudaMemcpyHostToDevice);

    sum_variance<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(device_input_vector, device_variance, device_mean, N);

    cudaMemcpy(variance, device_variance, sizeof(float), cudaMemcpyDeviceToHost);

    *variance /= N;

    cudaMemcpy(device_variance, variance, sizeof(float), cudaMemcpyHostToDevice);

    layer_norm<<<blocks_per_grid, threads_per_block>>>(device_input_vector, device_output_vector, device_mean, device_variance, N);

    cudaMemcpy(host_output_vector, device_output_vector, size, cudaMemcpyDeviceToHost);

    check_layer_norm(host_input_vector, host_output_vector, iterative_output_vector, N, tolerance);


    cudaFree(device_input_vector); cudaFree(device_output_vector); cudaFree(device_mean); cudaFree(device_variance);
    free(host_input_vector); free(host_output_vector); free(iterative_output_vector); free(mean); free(variance);

    return 0;
}