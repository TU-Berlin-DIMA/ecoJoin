#include <iostream>
#include <unistd.h>

#define CUDA_SAFE(func)    {                                              \
        cudaError err = func;                                             \
        if (cudaSuccess != err) {                                         \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

__global__ 
void kernel(int *mutex){
	while (true){
		while (*mutex == 0){}
		*mutex = 2; //atomic inc
	}
}

int main(){
	int *mutex;
	CUDA_SAFE(cudaHostAlloc(&mutex, sizeof(int),0));
	*mutex = 0;

	kernel<<<1,10>>>(mutex);
	*mutex = 1;
	while(*mutex == 1){} // < 11
	std::cout << "mutex is "<< *mutex << "\n";
}
