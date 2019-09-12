#include <chrono>
#include <iostream>
#include <omp.h>
#include "benchmark_helper.h"

 #define CUDA_SAFE(call)                                                 \
    do {                                                                \
    cudaError_t err = call;                                             \
    if (cudaSuccess != err) {                                           \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.",     \
            __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
    } while (0)

#define START_M(name) \
    std::chrono::time_point<std::chrono::system_clock> start_name, end_name; \
    start_name = std::chrono::system_clock::now();

#define END_M(name) \
    end_name = std::chrono::system_clock::now(); \
    long elapsed_seconds_name = std::chrono::duration_cast<std::chrono::nanoseconds> (end_name-start_name).           count(); \
    //std::cout << elapsed_seconds_name << "\n";

#define runs_ 1
__global__
void bench(int *in, int  *out, int size){
	unsigned global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t global_threads = blockDim.x * gridDim.x;
    for (
            unsigned tuple_id = global_idx;
            tuple_id < size;
            tuple_id += global_threads
        )
    {
		out[tuple_id] = in[tuple_id];
    }
}


int main(){
	int k = 45000;
	long n = 4096 *k;

	int *in_h, *out_h;
	int *in_d, *out_d;

//	cudaHostAlloc((void**)&in_h,  n*sizeof(int), cudaHostAllocMapped);;
//	cudaHostAlloc((void**)&out_h, n*sizeof(int), cudaHostAllocMapped);;
	
	cudaHostAlloc((void**)&in_h,  n*sizeof(int), cudaHostAllocDefault);;
	cudaHostAlloc((void**)&out_h, n*sizeof(int), cudaHostAllocDefault);;
	

	CUDA_SAFE(cudaMalloc((void**)&in_d,  n*sizeof(int)));
	CUDA_SAFE(cudaMalloc((void**)&out_d, n*sizeof(int)));

	//cudaMallocManaged((void**)&in_h,  n*sizeof(int));
	//cudaMallocManaged((void**)&out_h, n*sizeof(int));
	

	int streamNumber = 5;
    cudaStream_t *streams = (cudaStream_t *)malloc(streamNumber * sizeof(cudaStream_t));
    for (int i = 0; i < streamNumber; i++)
      CUDA_SAFE(cudaStreamCreate(&streams[i]));

    int block = 1024 * 1024 *128; //  Blocksize in byte
    block = 1024 * 1024 * 32; //  Blocksize in byte
    long sizeByte =  n* sizeof(int);
    INIT_ASYNC(block, int)


	double gb = (double) n* sizeof(int) / 1024 / 1024 / 1024 * runs_;
	long total = 0;

	int runs = 1;
	for (int j = 0; j < runs; j++){
		
		for (long i = 0; i < n; i++){
			in_h[i] = 1;
			out_h[i] = 0;
		}

        START_M()
#pragma omp parallel for
		for (int k = 0; k < runs_; k++) {
			for (int i = 0; i < blockNumber; i++) {
				int offset = i * elementsPerBlock;
				CUDA_SAFE(cudaMemcpyAsync(&in_d[offset], &in_h[offset], blockByte,
										cudaMemcpyHostToDevice,
										streams[currentStream]));
				bench<<<128*32,64, 0, streams[currentStream]>>>(&in_d[offset], &out_d[offset], blockByte/sizeof(int));
				CUDA_SAFE(cudaMemcpyAsync(&out_h[offset], &out_d[offset],
									blockByte, cudaMemcpyDeviceToHost,
									streams[currentStream]));
				currentStream = ++currentStream % streamNumber;
			}
		}
        CUDA_SAFE(cudaDeviceSynchronize());
        END_M()
		
		std::cout << "blockNumer: " << blockNumber  << " ElementPerBlock: " << blockByte/sizeof(int) << std::endl;
		total += elapsed_seconds_name;
		//for (int i = 0; i < n; i++)
		//	if (out_h[i] != 1) printf("Error\n");

	}
	double sec  = (double)(total / runs) / 1000000000;

	std::cout << "sec " << sec << " gb " << gb << " gb/s " << gb/sec<<"\n";

}
