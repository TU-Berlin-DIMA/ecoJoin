#include <iostream>
#include <unistd.h>


#define WAIT 0
#define START 1


#define CUDA_SAFE(func)    {                                              \
        cudaError err = func;                                             \
        if (cudaSuccess != err) {                                         \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

const int table_size = 512; // shared mem max

 __device__
void build_linprobe(int key, int val, int *content, int tuple_id) {
    uint64_t location = key;

	//location *= HASH_FACTOR;
	//location &= table_size;
	//location = location % table_size;
	//location = (13 ^ location ) & table_size;
	location &= (table_size-1);

	for (int j = 0; ; j++) {
		int current_key = content[location * 2];
		if (current_key == 0) {
			int old = atomicCAS(&content[location * 2], 0, key);
			if (old == 0) {
				content[location * 2 + 1] = val;
				break;
			}
		}
		if ((++location)*2 == table_size)
			location = 0;
	}
}


__global__ 
void kernel(int *mutex, int *new_val, int *new_key){
    long  global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    extern __shared__ int content[];

	while (true){
		while (*mutex == WAIT){}
		
		build_linprobe(*new_key, *new_val, content, global_idx);

		*mutex = WAIT;
	}
}

int main(){
	int *mutex, *new_key, *new_val;
	CUDA_SAFE(cudaHostAlloc(&mutex,  sizeof(int),0));
	CUDA_SAFE(cudaHostAlloc(&new_val, sizeof(int),0));
	CUDA_SAFE(cudaHostAlloc(&new_key, sizeof(int),0));
	*mutex   = 0;
	*new_key = 0;
	*new_val = 0;

	kernel<<<1,1, 2*512*sizeof(int)>>>(mutex, new_key, new_val);

	// New tuple arrives
	*mutex = START;
	while(*mutex == START){}
	std::cout << "mutex is "<< *mutex << "\n";
}
