/*
 *  Helper lib for benchmarking cuda
 *
 *  FILE: benchmark_helper_lib.cu
 */

// TODO: Calc Bandwidth

#include <stdio.h>
#include <stdlib.h>

#define CUDA_SAFE(func)                                                   \
    {                                                                     \
        cudaError err = func;                                             \
        if (cudaSuccess != err) {                                         \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }


void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop) {
    CUDA_SAFE(cudaEventCreate(start))
    CUDA_SAFE(cudaEventCreate(stop))
    CUDA_SAFE(cudaEventRecord(*start, 0));
}

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop) {
    CUDA_SAFE(cudaEventRecord(stop));
    CUDA_SAFE(cudaEventSynchronize(start));
    CUDA_SAFE(cudaEventSynchronize(stop));
    float kernel_time;
    CUDA_SAFE(cudaEventElapsedTime(&kernel_time, start, stop));
    CUDA_SAFE(cudaEventDestroy(start));
    CUDA_SAFE(cudaEventDestroy(stop));
    return kernel_time;
}

// float  bandwidth =
// ((double)memoryoperations*sizeof(datatype))/kernel_time*1000./(1000.*1000.*1000.);
// float getBandwidth(){
//
//}
__global__ void increment(float *array_d, unsigned N) {
     unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < N) {
         array_d[idx]++;
     }
     //else {
     //   printf("failed %d, %d", idx, N);
     //}
 }

template <typename T>
__global__ void increment(T *array_d, unsigned N) {
     unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < N) {
         array_d[idx]++;
     }
     //else {
     //   printf("failed %d, %d", idx, N);
     //}
 }


bool checkIncrementResult(unsigned *input, unsigned *output, unsigned sizeInt,
                          unsigned iterations) {
    bool ok = true;
    for (unsigned k = 0; k < sizeInt - 1; k++) {
        if ((input[k] + iterations) != output[k]) {
            ok = false;
        }
    }
    if (ok) {
        printf("test failed\n");
        printf("number of elements: %d\n", sizeInt);
        for (unsigned k = 0; k < 100; k++) {
            printf("%d, ", output[k]);
        }
        printf("\n");
    }
    return ok;
}

bool checkIncrementResult(unsigned *input, unsigned *output, unsigned sizeInt) {
    bool ok = true;
    for (unsigned k = 0; k < sizeInt - 1; k++) {
        if ((input[k] + 1) != output[k]) {
            ok = false;
        }
    }
    if (ok) {
        printf("test failed\n");
        printf("number of elements: %d\n", sizeInt);
        for (unsigned k = 0; k < 100; k++) {
            printf("%d, ", output[k]);
        }
        printf("\n");
    }
    
    return ok;
}

//void printResults(char* method, ,long long computations, float ms){
//    printf("Kernel Execution Time: %10.3f msec\n");
//    printf("            Bandwidth: ", computaions*sizeof(datatype)/(ms/1000)/);
//}
