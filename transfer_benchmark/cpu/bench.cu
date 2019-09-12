#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

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
        //out[in[tuple_id] % size] = in[tuple_id];
        out[tuple_id] = in[tuple_id];
    }
}

int main(){
	int k = 10000;
	long n = 4096 *k;

	int *in_h, *out_h;
	int *in_d, *out_d;

	cudaHostAlloc((void**)&in_h,  n*sizeof(int), cudaHostAllocDefault);;
	cudaHostAlloc((void**)&out_h, n*sizeof(int), cudaHostAllocDefault);;
	
	CUDA_SAFE(cudaMalloc((void**)&in_d,  n*sizeof(int)));
	CUDA_SAFE(cudaMalloc((void**)&out_d, n*sizeof(int)));

	//cudaMallocManaged((void**)&in_h,  n*sizeof(int));
	//cudaMallocManaged((void**)&out_h, n*sizeof(int));
	
	double gb = (double) n* sizeof(int) / 1024 / 1024 / 1024 * runs_;
	long total = 0;

	int runs = 1;
	for (int j = 0; j < runs; j++){
		
		// (1)
		for (long i = 0; i < n; i++){
			in_h[i] = i;
			out_h[i] = i;
		}

		// (2)
		/*std::vector<int> tmp1, tmp2;
        tmp1.reserve(n);
        tmp2.reserve(n);
        double sel = 0.03;
        for (int i = 0; i < n; i++){
            tmp1[i] = i+1 + (n - n*sel);
            tmp2[i] = i+1;
        }
		std::vector<int> indexes1;
		std::vector<int> indexes2;
		indexes1.reserve(n);
		indexes2.reserve(n);
		for (int i = 0; i < n; ++i){
		    indexes1.push_back(i);
		    indexes2.push_back(i);
		}
		std::random_shuffle(indexes1.begin(), indexes1.end());
		std::random_shuffle(indexes2.begin(), indexes2.end());
		for (int i = 0; i < n; i++){
		    in_h[i] = tmp1[indexes1[i]];
		    out_h[i] = tmp2[indexes2[i]];
		}
		indexes1.clear();
		indexes2.clear();
		tmp1.clear();
		tmp2.clear();
		*/

		START_M()
#pragma omp parallel for 
		for (int i= 0; i < n; i++){
			out_h[i] = in_h[i];
		}
		END_M()
		
		CUDA_SAFE(cudaMemcpy(in_h, in_d, n* sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_SAFE(cudaMemcpy(out_h, out_d, n* sizeof(int), cudaMemcpyDeviceToHost));

		total += elapsed_seconds_name;
		//for (int i = 0; i < n; i++)
		//	if (out_h[i] != 1) printf("Error\n");

	}
	double sec  = (double)(total / runs) / 1000000000;

	std::cout << "sec " << sec << " gb " << gb << " gb/s " << (2*gb)/sec<<"\n";
}
