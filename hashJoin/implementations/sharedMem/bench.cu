#include <chrono>
#include <iostream>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "hashtbl.h"

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


void parseCSV(std::string filename, int *keys, int *vals){
    std::ifstream file(filename);
    std::string line;
    int row = 0;
    while (std::getline(file, line)){
        std::stringstream iss(line);
        std::string key, val;
        std::getline(iss, key , ',');
        std::getline(iss, val , ',');
		keys[row] = std::stoi(key);
		vals[row] = std::stoi(val);
        row++;
    }
    std::cout << filename << ": " << row << "rows loaded" << std::endl;
};


int main() {
	std::string probe_file = "/home/adimon/DBAPRO/simpleStreamingJoin/dataset/sel003/probe_1000000.csv"; 
	std::string build_file = "/home/adimon/DBAPRO/simpleStreamingJoin/dataset/sel003/build_1000000.csv"; 

	int k = std::pow(2,14);
	unsigned n = k;// 4096 *k;
	unsigned c = n;//n * 2;

	/* ------------------------------------------------------------------------*/
	/* ------------------------------------------------------------------------*/

	int *vals_p_h, *keys_p_h, *content;
	int *vals_b_h, *keys_b_h;

	// (1)
	CUDA_SAFE(cudaHostAlloc((void**)&keys_p_h,  n*sizeof(int), cudaHostAllocDefault));
	CUDA_SAFE(cudaHostAlloc((void**)&vals_p_h,  n*sizeof(int), cudaHostAllocDefault));

	CUDA_SAFE(cudaHostAlloc((void**)&keys_b_h,  n*sizeof(int), cudaHostAllocDefault));
	CUDA_SAFE(cudaHostAlloc((void**)&vals_b_h,  n*sizeof(int), cudaHostAllocDefault));
	
	// (2)
	//cudaMalloc((void**)&keys_h,   n*sizeof(unsigned));
	//cudaMalloc((void**)&vals_h,   n*sizeof(unsigned));
	// (3)
	//CUDA_SAFE(cudaMallocManaged((void**)&keys_h,  n*sizeof(unsigned)));
	//CUDA_SAFE(cudaMallocManaged((void**)&vals_h,  n*sizeof(unsigned)));
	//CUDA_SAFE(cudaMallocManaged((void**)&keys_h,  n*sizeof(int32_t)));
	//CUDA_SAFE(cudaMallocManaged((void**)&vals_h,  n*sizeof(int32_t)));

	// (1)
	CUDA_SAFE(cudaMalloc((void **)&content, 2*c*sizeof(int)));
	//cudaMalloc((void **)&content, 2*c*sizeof(int32_t));
	// (2)
	//cudaHostAlloc((void **)&content, 2*c*sizeof(unsigned), cudaHostAllocDefault);
	// (3)
	//CUDA_SAFE(cudaMallocManaged((void**)&content,  2*c*sizeof(unsigned)));
	
	for (int i = 0; i < n; i++){
		keys_p_h[i] = i;
		vals_p_h[i] = i;
	}

	//parseCSV(probe_file, keys_p_h, vals_p_h);
	//parseCSV(build_file, keys_b_h, vals_b_h);

	/* ------------------------------------------------------------------------*/
	/* ------------------------------------------------------------------------*/

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));

	double gb = (double) n*2*sizeof(int) / 1024 / 1024 / 1024;
	long total = 0;

	int runs = 1;
	for (int j = 0; j < runs; j++){
		printf("Run %d\n",j);
		
		CUDA_SAFE(cudaMemset(content, 0, 2*c*sizeof(int)));

		//START_M()

		CUDA_SAFE(cudaEventRecord(start));
		build_linprobe_shared<<<k / 1024,128, n*sizeof(int),stream>>>(n, content, c,  keys_p_h, vals_p_h);
		CUDA_SAFE(cudaEventRecord(stop));

		CUDA_SAFE(cudaEventSynchronize(stop));
		float milliseconds = 0;
		CUDA_SAFE(cudaEventElapsedTime(&milliseconds, start, stop));

		CUDA_SAFE(cudaStreamSynchronize(stream));
		//END_M()
		
		int *content_h;
		cudaHostAlloc((void**)&content_h,  2*c*sizeof(int), cudaHostAllocDefault);;
		CUDA_SAFE(cudaMemcpy(content_h, content, 2*c*sizeof(int),cudaMemcpyDeviceToHost));

		for (int i = 0; i < c; i++)
			if (content_h[i*2] == 0) printf("Error %d, %d\n",i, content_h[2*i]);
		
		total += milliseconds;

	}
	double sec  = (double)(total / runs) / 1000;
	std::cout << "sec " << sec << " gb " << gb << " gb/s " << gb/sec<<"\n";
}
