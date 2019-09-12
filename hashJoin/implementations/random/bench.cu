#include <chrono>
#include <thread>
#include <iostream>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>

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

int main() {
	int k = pow(2,23); // Input size
	unsigned n = k; // Entries
	unsigned buckets = n * 2 ; // Table Size (2* entries)

	/* ------------------------------------------------------------------------*/
	/* ------------------------------------------------------------------------*/

	int *vals_p_h, *keys_p_h, *content;
	int *vals_b_h, *keys_b_h, *current_d, *output_h;

	// (1)
	cudaHostAlloc((void**)&keys_p_h,  n*sizeof(int), cudaHostAllocDefault);;
	cudaHostAlloc((void**)&vals_p_h,  n*sizeof(int), cudaHostAllocDefault);;
	cudaHostAlloc((void**)&keys_b_h,  n*sizeof(int), cudaHostAllocDefault);;
	cudaHostAlloc((void**)&vals_b_h,  n*sizeof(int), cudaHostAllocDefault);;
	
	// (2)
	//cudaMalloc((void**)&keys_h,   n*sizeof(unsigned));
	//cudaMalloc((void**)&vals_h,   n*sizeof(unsigned));
	// (3)
	//CUDA_SAFE(cudaMallocManaged((void**)&keys_b_h,  n*sizeof(int)));
	//CUDA_SAFE(cudaMallocManaged((void**)&vals_b_h,  n*sizeof(int)));

	// (1)
	CUDA_SAFE(cudaMalloc((void **)&content, 2*buckets*sizeof(int)));
	CUDA_SAFE(cudaMalloc((void **)&current_d, sizeof(int)));
	//cudaMalloc((void **)&content, 2*c*sizeof(int32_t));
	// (2)
	//cudaHostAlloc((void **)&content, 2*buckets*sizeof(int), cudaHostAllocDefault);
	// (3)
	//CUDA_SAFE(cudaMallocManaged((void**)&content,  2*buckets*sizeof(int)));

	std::vector<int> tmp1, tmp2;
	tmp1.reserve(n);
	tmp2.reserve(n);
	double sel = 0.03;
	for (int i = 0; i < n; i++){
		tmp1[i] = i+1 + (n - n*sel);
		tmp2[i] = i+1;
	}
	std::cout <<  "p range " << tmp1[0] << " " << tmp1[n-1]<< "\n";
	std::cout <<  "b range " << tmp2[0] << " " << tmp2[n-1]<< "\n";

	/* shuffle */
	std::vector<int> indexes1;
	std::vector<int> indexes2;
	indexes1.reserve(n);
	indexes2.reserve(n);
	for (int i = 0; i < n; ++i){
		indexes1.push_back(i);
		indexes2.push_back(i);
	}

	/* Comment this to remove shuffle*/
	std::random_shuffle(indexes1.begin(), indexes1.end()); //probe
	std::random_shuffle(indexes2.begin(), indexes2.end()); //build
	/* Comment this to remove shuffle*/

	for (int i = 0; i < n; i++){
		keys_p_h[i] = tmp1[indexes1[i]];
		vals_p_h[i] = tmp1[indexes1[i]];
		keys_b_h[i] = tmp2[indexes2[i]];
		vals_b_h[i] = tmp2[indexes2[i]];
	}

	indexes1.clear();
	indexes2.clear();
	int outputsize = (tmp2[n-1] - tmp1[0]+1)*3;
	std::cout << "Outputsize " << outputsize << "\n";
	CUDA_SAFE(cudaHostAlloc((void **)&output_h, outputsize*sizeof(int), cudaHostAllocDefault));
	//CUDA_SAFE(cudaHostAlloc((void **)&output_h, 2*buckets*sizeof(int), cudaHostAllocDefault));
	tmp1.clear();
	tmp2.clear();

	std::cout << "Integers in HT (4*k) " << 4*k << "\n";

	
	/* ------------------------------------------------------------------------*/
	/* ------------------------------------------------------------------------*/

	cudaEvent_t start_b, stop_b, start_p, stop_p;
	cudaEventCreate(&start_b);
	cudaEventCreate(&stop_b);
	cudaEventCreate(&start_p);
	cudaEventCreate(&stop_p);
	
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));

	double gb = (double) n*4*sizeof(int) /1024/1024/1024;
	long total_b = 0;
	long total_p = 0;

	//int runs =400;
	int runs =30;
	for (int j = 0; j < runs; j++){
		printf("Run %d\n",j);
		
		std::memset(output_h, 0, outputsize*sizeof(int));
		CUDA_SAFE(cudaMemset(content, 0, 2*buckets*sizeof(int)));
		CUDA_SAFE(cudaMemset(current_d, 0, sizeof(int)));
	
		{	
		START_M()
		build_linprobe_lsb<<<32*128,1024,0,stream>>>(n, content, buckets,  keys_b_h, vals_b_h);
		CUDA_SAFE(cudaStreamSynchronize(stream));
		END_M()
		total_b += elapsed_seconds_name;
		}
		
		{
		START_M()
		probe_linprobe_lsb<<<32*128,1024,0,stream>>>(n, keys_p_h, vals_p_h, content, buckets, output_h, current_d);
		CUDA_SAFE(cudaStreamSynchronize(stream));
		END_M()
		total_p += elapsed_seconds_name;
		}
		// Error check
		/*
		int *content_h;
		cudaHostAlloc((void**)&content_h,  2*c*sizeof(int), cudaHostAllocDefault);;
		CUDA_SAFE(cudaMemcpy(content_h, content, 2*c*sizeof(int),cudaMemcpyDeviceToHost));
		 */

		/* Content dumb */
		/*int *content_h;
		CUDA_SAFE(cudaHostAlloc((void **)&content_h, 2*buckets*sizeof(int), cudaHostAllocDefault));
		CUDA_SAFE(cudaMemcpy(content_h, content, 2*buckets*sizeof(int),cudaMemcpyDeviceToHost));
		for (int i = 0; i < buckets; i++)
			printf("co %d, %d\n", content_h[i*2], content_h[i*2+1]);
		*/

		/* Output Check*/
		int current_h;
		CUDA_SAFE(cudaMemcpy(&current_h, current_d,sizeof(int),cudaMemcpyDeviceToHost));
		std::cout << "current: " <<current_h << "\n";
		/*for (int i = 0; i < current_h*3; i++){
			if (output_h[i*3] != 0) {
				printf("output %d, %d, %d\n",output_h[i*3], output_h[i*3+1], output_h[i*3+2]);
			}
		}*/
	}
	double nsec_b  = (double)(total_b / runs);
	double nsec_p  = (double)(total_p / runs);
	std::cout << "Build: nsec " << nsec_b << " gb " << gb << " gb/s " << (4*n*sizeof(int))/nsec_b<<"\n";
	std::cout << "Probe: nsec " << nsec_p << " gb " << gb << " gb/s " << (4*n*sizeof(int))/nsec_p<<"\n";
	std::cout << "EtoE:  nsec " << nsec_b+nsec_p << " gb " << gb << " gb/s " << (4*n*sizeof(int))/(nsec_p+nsec_b)<<"\n";
}
