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
    std::chrono::time_point<std::chrono::system_clock> start_name, end_name; \
    start_name = std::chrono::system_clock::now();

#define END_M(name) \
    end_name = std::chrono::system_clock::now(); \
    long elapsed_seconds_name = std::chrono::duration_cast<std::chrono::nanoseconds> (end_name-start_name).           count(); \
    //std::cout << elapsed_seconds_name << "\n";

__global__
void bench(int *in, int  *out, int e, int v){
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < v;i++)
		out[idx+e*i] = in[idx+e*i];
}

int main(){
	int v = 4;
	int k = 90000;
	long n = 4096 *k;

	long elmProBlock = 4096*k/v;

	int *in_h, *out_h;
	int *in_d, *out_d;

	cudaHostAlloc((void**)&in_h,  n*sizeof(int), cudaHostAllocMapped);;
	cudaHostAlloc((void**)&out_h, n*sizeof(int), cudaHostAllocMapped);;
	
	double gb = (double) n* sizeof(int) / 1024 / 1024 / 1024;
	long total = 0;

	int runs = 2;
	for (int j = 0; j < runs; j++){
		
		for (long i = 0; i < n; i++){
			in_h[i] = 1;
			out_h[i] = 0;
		}

		START_M()
		bench<<<elmProBlock/1024, 1024>>>(in_h, out_h,elmProBlock, v);
		cudaDeviceSynchronize();
		END_M()

		total += elapsed_seconds_name;
		for (int i = 0; i < n; i++)
			if (out_h[i] != 1) printf("Error\n");

	}
	double sec  = (double)(total / runs) / 1000000000;

	std::cout << "sec " << sec << " gb " << gb << " gb/s " << gb/sec<<"\n";
}
