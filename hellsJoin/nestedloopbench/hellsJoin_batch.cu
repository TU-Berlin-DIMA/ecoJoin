#include <bitset>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mutex>
#include <vector>
#include <chrono>
#include <thread>
#include <cstring>

#define MANAGED 0
#define ZERO_COPY_R 0

#define DEBUG_P(pr) if (DEBUG) std::cout << pr << "\n";
//#define DEBUG_P(pr) std::cout << pr << "\n";

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
	int elapsed_seconds_name = std::chrono::duration_cast<std::chrono::milliseconds> (end_name-start_name).count(); \
    std::time_t end_time_name = std::chrono::system_clock::to_time_t(end_name);\
    std::cout << "elapsed time: " << elapsed_seconds_name << "ms\n";\

enum Stream { stream1, stream2 };

struct tuple {
    int key;
    int value;
};

struct record2 {
    int key;
    int left_value;
    int right_value;
};

tuple *compareTuples_s1_inter, *compareTuples_s1_comp;
__device__ size_t currentFIFO_s1 = 0;
size_t currentFIFO_s1_inter = 0;
tuple *compareTuples_s2_inter, *compareTuples_s2_comp;
__device__ size_t currentFIFO_s2 = 0;
size_t currentFIFO_s2_inter = 0;

std::vector<tuple> compareTuples_h;
int *compare_output_s1,  *compare_output_s2;
int *compare_output_prev;

int etpw, gridsize, blocksize;

std::ofstream myfile;

int batchsize;

void printRecord(tuple rec){
	DEBUG_P("key: " << rec.key <<" value: " << rec.value)
}

__global__
void nested_loop(tuple *input, tuple *compareTuples, tuple *output, int etpw, int *current){
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	long  global_threads = blockDim.x * gridDim.x;
	for (long tuple_id = idx; tuple_id < etpw; tuple_id += global_threads){
        for (int j = 0; j < etpw; j++){ // work on batch
            if  (input[j].key == compareTuples[tuple_id].key){
				//atomicAdd(current, 1);
				output[tuple_id].key = 1; //input[j].key;
			}
        }
    }	
}

void startFileTest(){
	for (int i = 10; i < 24; i++){
		std::cout << i << "\n";
		int n = std::pow(2,i);
		tuple *records0;
		tuple *records1;
		tuple *output;
		int *current;
		CUDA_SAFE(cudaHostAlloc((void **)&records0, sizeof(tuple) * n, 0));
		CUDA_SAFE(cudaHostAlloc((void **)&records1, sizeof(tuple) * n, 0));
		
		CUDA_SAFE(cudaHostAlloc((void **)&output, sizeof(tuple) * n, 0));
		CUDA_SAFE(cudaMalloc((void **)&current, sizeof(int)));

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
	
		/* Comment this to remove shuffle */
		//std::random_shuffle(indexes1.begin(), indexes1.end()); //probe
		//std::random_shuffle(indexes2.begin(), indexes2.end()); //build
		/* Comment this to remove shuffle*/
		
		for (int i = 0; i < n; i++){
			records0[i].key = tmp1[indexes1[i]];
			records0[i].value= tmp1[indexes1[i]];
			records1[i].key = tmp2[indexes1[i]];
			records1[i].value= tmp2[indexes1[i]];
		}
	
		indexes1.clear();
		indexes2.clear();
		int outputsize = (tmp2[n-1] - tmp1[0]+1)*3;
		std::cout << "Outputsize " << outputsize << "\n";
		//CUDA_SAFE(cudaHostAlloc((void **)&output_h, outputsize*sizeof(int), cudaHostAllocDefault));
		//CUDA_SAFE(cudaHostAlloc((void **)&output_h, 2*buckets*sizeof(int), cudaHostAllocDefault));
		tmp1.clear();
		tmp2.clear();
		
		//parseCSV(filename1.c_str(), records0);
		//parseCSV(filename2.c_str(), records1);
		
		START_M(_)
		nested_loop<<<128*32, 1024>>>(records0, records1, output,n, current);
		CUDA_SAFE(cudaStreamSynchronize(0));
		END_M(_)
		double gb = (double) (n*sizeof(int)*4) / 1024 / 1024 / 1024;
		double s  = (double) elapsed_seconds_name / 1000;
		std::cout << "gb "<< gb << " sec " << s << " gb/s " << gb/s << "\n";

		int current_h = 0;
		CUDA_SAFE(cudaMemcpy(&current_h, current, sizeof(int), cudaMemcpyDeviceToHost));
		std::cout << current_h << "\n";

	}
    /*START_M(_)	
	int i = 0;
	int j = 0;
	Stream stream_prev;
	tuple  *tuple_prev;
	tuple_prev = (tuple *) malloc(sizeof(tuple)*batchsize);
	tuple  new_tuple;

    for (int k = 0; i < rows  && j < rows; k++) {
		while (i < rows) {
			compare_kernel_ipt<<<128, 128>>>(&records0[i], batchsize, compare_output_s1,  etpw, compareTuples_s2_comp);
			for (int z = 0; z < batchsize; z++)
				add_new_tuple_device<<<1, 1>>>(records0[i+z], stream1, etpw, compareTuples_s1_comp);
			

			// Start interpretation of prev tuple while execution
			int z = 0;
			if (i!=0)
				//for (int z = 0; z < batchsize; z++)
					write_result(interprete(tuple_prev[z], &compare_output_prev[z*((etpw / 32) + 1)], stream_prev));
			
			CUDA_SAFE(cudaStreamSynchronize(0));

			// Save prev setup.
			std::memcpy(compare_output_prev, compare_output_s1, sizeof(int) * ((etpw / 32) + 1) * batchsize);
			stream_prev = stream1;
			std::memcpy(tuple_prev, &records0[i], sizeof(tuple)*batchsize);

			i = i + batchsize;
			if (((i+j) % 10000) == 0) printf("%d\n", i+j);
		}
		while (j < rows) {
			compare_kernel_ipt<<<128, 128>>>(&records1[j], batchsize, compare_output_s2,  etpw, compareTuples_s1_comp);
			for (int z = 0; z < batchsize; z++)
				add_new_tuple_device<<<1, 1>>>(records1[j+z], stream2, etpw, compareTuples_s2_comp);
			
			// Start interpretation of prev tuple while execution
			int z = 0;
			if (j!=0)
				//for (int z = 0; z < batchsize; z++)
					write_result(interprete(tuple_prev[z], &compare_output_prev[z*((etpw / 32) + 1)], stream_prev));
			
			CUDA_SAFE(cudaStreamSynchronize(0));

			std::memcpy(compare_output_prev, compare_output_s2, sizeof(int) * ((etpw / 32) + 1) * batchsize);
			stream_prev = stream2;
			std::memcpy(tuple_prev, &records1[j], sizeof(tuple)*batchsize);

			j = j + batchsize;
			if (((i+j) % 10000) == 0) printf("%d\n", i+j);
		}
    }
	END_M(_)	*/
}

int main(int argc, char *argv[]){

	startFileTest(); // Filename1, Filename2, rows
}
