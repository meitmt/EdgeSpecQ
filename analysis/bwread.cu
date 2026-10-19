// Read-bandwidth ceiling on this GPU: stream a buffer once per launch with 16-byte loads, sum to keep the loads.
// nvcc -O3 -arch=sm_87 bwread.cu -o bwread ; ./bwread [MiB=117] [iters=50]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void read_kernel(const float4 * __restrict__ p, size_t n, float * out) {
    float acc = 0.0f;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        float4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) {
        out[0] = acc;  // practically never true, keeps the loads alive
    }
}

int main(int argc, char ** argv) {
    const double mib = argc > 1 ? atof(argv[1]) : 117.0;
    const int iters = argc > 2 ? atoi(argv[2]) : 50;
    const size_t bytes = (size_t) (mib * 1024 * 1024) / 16 * 16;
    const size_t n = bytes / 16;
    float4 * p; float * out;
    cudaMalloc(&p, bytes); cudaMalloc(&out, 4);
    cudaMemset(p, 1, bytes);
    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("%s, %d SMs, L2 %d KiB, buffer %.1f MiB\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize / 1024, bytes / 1048576.0);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int blocks_per_sm : {4, 8, 16, 32}) {
        const int grid = prop.multiProcessorCount * blocks_per_sm;
        read_kernel<<<grid, 256>>>(p, n, out);  // warm-up
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) read_kernel<<<grid, 256>>>(p, n, out);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        printf("read  grid %4d x 256: %7.1f us/iter  %6.1f GB/s\n", grid, 1e3 * ms / iters, bytes * (double) iters / (ms * 1e-3) / 1e9);
    }
    float4 * q; cudaMalloc(&q, bytes);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) cudaMemcpy(q, p, bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    printf("memcpy D2D (read + write): %6.1f GB/s moved\n", 2.0 * bytes * iters / (ms * 1e-3) / 1e9);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 0;
}
