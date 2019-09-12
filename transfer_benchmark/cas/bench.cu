#include <chrono>
#include <iostream>

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
    std::chrono::time_point<std::chrono::steady_clock> start_name, end_name; \
    start_name = std::chrono::steady_clock::now();

#define END_M(name) \
    end_name = std::chrono::steady_clock::now(); \
    unsigned long long elapsed_seconds_name = std::chrono::duration_cast<std::chrono::nanoseconds> (end_name-start_name).count(); \
    //std::cout << elapsed_seconds_name << "\n";

#define runs_ 1
__global__
void bench(unsigned long long *in, unsigned long long  *out, int size){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_threads = blockDim.x * gridDim.x;
    for (
            int tuple_id = global_idx;
            tuple_id < size;
            tuple_id += global_threads
        )
    {
        //out[tuple_id] = in[tuple_id];
		atomicCAS((unsigned long long*)&out[tuple_id],(unsigned long long)tuple_id, (unsigned long long) in[tuple_id]);
    }
}

int main(){
	int k = 45000 / 2;
	long n = 4096 *k;
	// long long n = 1024ll * 1024 * 1024 / sizeof(unsigned long long);

	unsigned long long *in_h, *out_h;
	unsigned long long *in_d, *out_d;

	cudaHostAlloc((void**)&in_h,  n*sizeof(unsigned long long), cudaHostAllocDefault);;
	cudaHostAlloc((void**)&out_h, n*sizeof(unsigned long long), cudaHostAllocDefault);;


	CUDA_SAFE(cudaMalloc((void**)&in_d,  n*sizeof(unsigned long long)));
	CUDA_SAFE(cudaMalloc((void**)&out_d, n*sizeof(unsigned long long)));

	// cudaMallocManaged((void**)&in_h,  n*sizeof(unsigned long long));
	// cudaMallocManaged((void**)&out_h, n*sizeof(unsigned long long));
	
	double  gb = (double) n* sizeof(unsigned long long) / 1024 / 1024 / 1024 * runs_;
	unsigned long long  total = 0;

	int runs = 20;
	for (int j = 0; j < runs; j++){
	/*	
		for (long i = 0; i < n; i++){
			in_h[i] = 1;
			out_h[i] = 0;
		}
*/
		for (int i = 0; i < n; i++) {
			in_h[i] = i;
			out_h[i] = i;
		}
	
		CUDA_SAFE(cudaMemcpy(in_d, in_h, n* sizeof(unsigned long long), cudaMemcpyHostToDevice));
		CUDA_SAFE(cudaMemcpy(out_d, out_h, n* sizeof(unsigned long long), cudaMemcpyHostToDevice));

		START_M()
		bench<<<128 * 32, 1024>>>(in_d, out_d, n);
		//bench_single<<<n/1024, 1024>>>(in_d, out_d);
		//bench<<<4*k, 1024>>>(in_h, out_h);
		CUDA_SAFE(cudaDeviceSynchronize());
		END_M()
		
		CUDA_SAFE(cudaMemcpy(in_h, in_d, n* sizeof(unsigned long long), cudaMemcpyDeviceToHost));
		CUDA_SAFE(cudaMemcpy(out_h, out_d, n* sizeof(unsigned long long), cudaMemcpyDeviceToHost));

		total += elapsed_seconds_name;
		/*for (unsigned long long i = 0; i < n; i++)
			if (out_h[i] != 1) prunsigned long longf("Error\n");*/

	}
	//unsigned long long sec  = (total / runs) / (1000ull *  1000ull * 1000ull);
	double  nsec  = (total / runs);

	//std::cout << "sec " << sec << " gb " << gb << " gb/s " << gb/sec<<"\n";
	std::cout << "nsec " << nsec << " gb " << gb << " gb/s " << (2 * n*sizeof(unsigned long long))/nsec<<"\n";
}
