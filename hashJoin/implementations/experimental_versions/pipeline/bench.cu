#include <chrono>
#include <iostream>
#include <cstring>

#include "hashtbl.h"
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


int main() {
	int k = 20000;
	unsigned n = 4096 *k;
	unsigned c = n * 2;

	unsigned *vals_h, *keys_h, *content, *vals_d, *keys_d;
	//int32_t *vals_h, *keys_h, *content;
	
	// (1)
	//cudaHostAlloc((void**)&keys_h,  n*sizeof(int32_t), cudaHostAllocDefault);;
	//cudaHostAlloc((void**)&vals_h,  n*sizeof(int32_t), cudaHostAllocDefault);;
	//cudaHostAlloc((void**)&keys_h,  n*sizeof(unsigned), cudaHostAllocDefault);;
	//cudaHostAlloc((void**)&vals_h,  n*sizeof(unsigned), cudaHostAllocDefault);;
	// (2)
	cudaHostAlloc((void**)&keys_h,  n*sizeof(unsigned), cudaHostAllocDefault);;
	cudaHostAlloc((void**)&vals_h,  n*sizeof(unsigned), cudaHostAllocDefault);;
	// (2)
	cudaMalloc((void**)&keys_d,   n*sizeof(unsigned));
	cudaMalloc((void**)&vals_d,   n*sizeof(unsigned));
	// (3)
	//CUDA_SAFE(cudaMallocManaged((void**)&keys_h,  n*sizeof(unsigned)));
	//CUDA_SAFE(cudaMallocManaged((void**)&vals_h,  n*sizeof(unsigned)));
	//CUDA_SAFE(cudaMallocManaged((void**)&keys_h,  n*sizeof(int32_t)));
	//CUDA_SAFE(cudaMallocManaged((void**)&vals_h,  n*sizeof(int32_t)));

	// (1)
	cudaMalloc((void **)&content, 2*c*sizeof(unsigned));
	//cudaMalloc((void **)&content, 2*c*sizeof(int32_t));
	// (2)
	//cudaHostAlloc((void **)&content, 2*c*sizeof(unsigned), cudaHostAllocDefault);
	// (3)
	//CUDA_SAFE(cudaMallocManaged((void**)&content,  2*c*sizeof(unsigned)));
	

	double gb = (double) n*2*sizeof(unsigned) / 1024 / 1024 / 1024;
	long total = 0;

	int runs = 5;
	for (int j = 0; j < runs; j++){
		printf("Run %d\n",j);
		
		//std::memset(content, 0, 2*c*sizeof(unsigned));

	/*	for (long i = 0; i < n; i++){
			keys_h[i] = i;
			vals_h[i] = i;
		}*/

		int streamNumber = 10;
		cudaStream_t *streams = (cudaStream_t *)malloc(streamNumber * sizeof(cudaStream_t));
        for (int i = 0; i < streamNumber; i++)
          CUDA_SAFE(cudaStreamCreate(&streams[i]));

        int block = 1024 * 1024 *128; //  Blocksize in byte
        block = 1024 * 1024 *16; //  Blocksize in byte
        long sizeByte =  n* sizeof(int);
        INIT_ASYNC(block, int)

        START_M()
		for (int i = 0; i < blockNumber; i++) { 
			int offset = i * elementsPerBlock; 
			CUDA_SAFE(cudaMemcpyAsync(&vals_d[offset], &vals_h[offset], blockByte, 
									cudaMemcpyHostToDevice, 
									streams[currentStream])); 
			CUDA_SAFE(cudaMemcpyAsync(&keys_d[offset], &keys_h[offset], blockByte, 
									cudaMemcpyHostToDevice, 
									streams[currentStream])); 
			build_linprobe<<<1,512, 0, streams[currentStream]>>>(n, content, c, &keys_d[offset], &vals_d[offset]); 
			//CUDA_SAFE(cudaMemcpyAsync(&keys_h[offset], &keys_d[offset], 
			//						blockByte, cudaMemcpyDeviceToHost, 
			//						streams[currentStream])); 
			//CUDA_SAFE(cudaMemcpyAsync(&vals_h[offset], &vals_d[offset], 
			//						blockByte, cudaMemcpyDeviceToHost, 
			//						streams[currentStream])); 
			currentStream = ++currentStream % streamNumber; 
		}
		CUDA_SAFE(cudaDeviceSynchronize());
        END_M()
		std::cout << "Blocknumber " << blockNumber << "\n";
		//build<<<4*k, 1024>>>(n, content, c,  keys_h, vals_h);
		//build_linprobe<<<k, 256>>>(n, content, c,  keys_h, vals_h);
		//build_linprobe<<<8,512>>>(n, content, c,  keys_h, vals_h);
		//gpu_ht_build_linearprobing_int32<<<40, 256>>>((int32_t *) content, (uint64_t)c,  keys_h, vals_h, (uint64_t)n);
		total += elapsed_seconds_name;
		//for (int i = 0; i < c; i++)
		//	if (content[i*2] != i) printf("Error %d, %d\n",i, content[2*i]);

	}
	double sec  = (double)(total / runs) / 1000000000;

	std::cout << "sec " << sec << " gb " << gb << " gb/s " << gb/sec<<"\n";
}
