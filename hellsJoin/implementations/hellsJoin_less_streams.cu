#include <bitset>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mutex>
#include <vector>
#include <queue>
#include <chrono>
#include <thread>
#include <cstring>
#include <omp.h>
#include "thread_pool.h"

#define MANAGED 0
#define ZERO_COPY_R 0

#define DEBUG_P(pr) if (DEBUG) std::cout << pr << "\n";

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
    int timestamp;
    int value;
};

struct record2 {
    int key;
    int timestamp;
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

struct queueElm{
public:
	queueElm(int *compare_output, Stream stream, tuple new_tuple)
		: compare_output(compare_output), stream(stream), new_tuple(new_tuple)
	{}
private:
	int *compare_output;
	Stream stream;
	tuple new_tuple;
};

std::queue<queueElm> queue;

void printRecord(tuple rec){
	DEBUG_P("key: " << rec.key << " timestamp:  "<< rec.timestamp << " value: " << rec.value)
}

/* int per thread
 * no race conditions
 * idx = etpw / 32
 */
__global__ 
void compare_kernel_ipt(tuple input, int *output, size_t etpw, tuple *compareTuples) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	int z = 0;
	if ((idx+1) * 32 < etpw){
#pragma unroll
		for (int i = 0; i < 32; i++) {
			if  ((input.timestamp == compareTuples[idx+(32*i)].timestamp)  // Time Window
			        && (input.key == compareTuples[idx+(32*i)].key)) {
//				printf("%d %d \n", input.key, input.timestamp);
				z = z | 1 << i;
			}
		}
    } else if (idx * 32 < etpw){
		for (int i = 0; i < etpw - idx*32 ; i++) {
			if  ((input.timestamp == compareTuples[idx+(32*i)].timestamp)  // Time Window
			        && (input.key == compareTuples[idx+(32*i)].key)) {
//				printf("%d %d \n", input.key, input.timestamp);
				z = z | 1 << i;
			}
		}
	}

	output[idx] = z;
}

/*
 *  Adds a new tuple to the device storage
 *  Start as one thread
 */
__global__ 
void add_new_tuple_device(tuple new_tuple, Stream stream, size_t etpw, tuple *compareTuples) {
	if (stream == stream1) {
		compareTuples[currentFIFO_s1].key = new_tuple.key;
		compareTuples[currentFIFO_s1].value = new_tuple.value;
		compareTuples[currentFIFO_s1].timestamp = new_tuple.timestamp;
		//printf("add_new_tuple_device %d %d at %d \n", new_tuple.timestamp, new_tuple.key, currentFIFO_s1);
		if(++currentFIFO_s1 == etpw)
			currentFIFO_s1 = 0;
	} else {
		compareTuples[currentFIFO_s2].key = new_tuple.key;
		compareTuples[currentFIFO_s2].value = new_tuple.value;
		compareTuples[currentFIFO_s2].timestamp = new_tuple.timestamp;
		//printf("add_new_tuple_device %d %d at %d \n", new_tuple.timestamp, new_tuple.key, currentFIFO_s2);
		if(++currentFIFO_s2 == etpw)
			currentFIFO_s2 = 0;
	}
}

__global__ 
void print_state(tuple *compareTuples, int etpw) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
		for (int i = 0; i < etpw; i++)
			printf("(%d, %d) ", compareTuples[i].key, compareTuples[i].timestamp);
		printf("\n");
    }
}

/*
 *  Interprete Bitmap as joinResult
 */
std::vector<record2> interprete(tuple input, int *bitmap, Stream stream) {
    DEBUG_P("Current rec ")
	printRecord(input);

	// Add locally
	if (stream == stream1) {
		compareTuples_s1_inter[currentFIFO_s1_inter].key = input.key;
		compareTuples_s1_inter[currentFIFO_s1_inter].value = input.value;
		compareTuples_s1_inter[currentFIFO_s1_inter].timestamp = input.timestamp;
		currentFIFO_s1_inter++;
		if(currentFIFO_s1_inter == etpw)
			currentFIFO_s1_inter = 0;
	} else {
		compareTuples_s2_inter[currentFIFO_s2_inter].key = input.key;
		compareTuples_s2_inter[currentFIFO_s2_inter].value = input.value;
		compareTuples_s2_inter[currentFIFO_s2_inter].timestamp = input.timestamp;
		currentFIFO_s2_inter++;
		if(currentFIFO_s2_inter == etpw)
			currentFIFO_s2_inter = 0;
	}

    std::vector<record2> result;
    for (int i = 0; i < etpw; i = i + 32) {
		if (bitmap[i / 32] == 0) { // first check
			continue;
		} else {
#pragma unroll
			for (int k = 0; k < 32; k++){
				int j = i+k;
				if (std::bitset<32>(bitmap[j / 32]).test(j % 32)) { // fine-grained check
					//bitmap[i / 32] = bitmap[j / 32] & ~(1 << (j % 32));  // ith bit = 0
					record2 r;
					if (stream == stream1) {
						DEBUG_P( "Match  ")
						printRecord(input);
						printRecord(compareTuples_s2_inter[j]);
						
						r.key = compareTuples_s2_inter[j].key;
						r.left_value = compareTuples_s2_inter[j].value;
						r.right_value = input.value;
						r.timestamp = compareTuples_s2_inter[j].timestamp;
					} else { 
						DEBUG_P("Match  ")
						printRecord(input);
						printRecord(compareTuples_s1_inter[j]);

						r.key = compareTuples_s1_inter[j].key;
						r.left_value = compareTuples_s1_inter[j].value;
						r.right_value = input.value;
						r.timestamp = compareTuples_s1_inter[j].timestamp;
					}
					result.push_back(r);
				}
			}
			bitmap[i / 32] = 0;
		}
    }
	free(bitmap);
	return result;
}

void print_result(std::vector<record2> result) {
    for (auto a : result)
		std::cout << "match  newtuple (" << a.timestamp << ", " << a.key << ", "
		<< a.right_value << ", " << a.left_value << ") \n";
}

void write_result(std::vector<record2> result){
    for (auto a : result)
		myfile << "match  newtuple (" << a.timestamp << ", " << a.key << ", "
		<< a.right_value << ", " << a.left_value << ") \n";
}

void parseCSV(std::string filename, tuple *tup){
	std::ifstream file(filename);
	std::string line;
	int row = 0;
	while (std::getline(file, line)){
		std::stringstream iss(line);
		std::string key, time, val;
		std::getline(iss, key , ',');
		std::getline(iss, time, ',');
		std::getline(iss, val , ',');

		tup[row] = {std::stoi(key), std::stoi(time), std::stoi(val)};
		row++;
	}
	std::cout << filename << ": " << row << "rows loaded" << std::endl;
};

void workertask(int *compare_output, Stream stream, tuple new_tuple){
}

void startFileTest(std::string filename1, std::string filename2, int rows){;
	tuple *records0 = new tuple[rows];
	tuple *records1 = new tuple[rows];

	parseCSV(filename1.c_str(), records0);
	parseCSV(filename2.c_str(), records1);
	
    START_M(_)	
	int i = 0;
	int j = 0;
	Stream stream_prev;
	tuple  tuple_prev;
	tuple  new_tuple;

	cudaStream_t cudastream1, cudastream2, cudastream3, cudastream4;
	cudaStreamCreate(&cudastream1);
	cudaStreamCreate(&cudastream2);
	cudaStreamCreate(&cudastream3);
	cudaStreamCreate(&cudastream4);
	
	ThreadPool pool(1);
	
    for (int k = 0; i < rows  && j < rows; k++) {
		while (records0[i].timestamp == k && i < rows) {
			new_tuple = records0[i];

			compare_kernel_ipt<<<blocksize, gridsize, 0, cudastream1>>>(new_tuple, compare_output_s1,  etpw, compareTuples_s2_comp);
			add_new_tuple_device<<<1, 1, 0, cudastream1>>>(new_tuple, stream1, etpw, compareTuples_s1_comp);

			//cudaStreamSynchronize(cudastream1);
			cudaStreamSynchronize(cudastream3);

			int *compare_output_new = (int *)malloc(((etpw / 32) + 1)* sizeof(int));
			std::memcpy(compare_output_new, compare_output_s1, sizeof(int) * ((etpw / 32) + 1));
			//queue.push(queueElm(compare_output_new, stream1, new_tuple));
			pool.enqueue([new_tuple, compare_output_new] (){write_result(interprete(new_tuple, compare_output_new, stream1));});

			i++;
			if (((i+j) % 10000) == 0) printf("%d\n", i+j);
		}
		while (records1[j].timestamp == k && j < rows) {
			new_tuple = records1[j];
			
			compare_kernel_ipt<<<blocksize, gridsize, 0, cudastream3>>>(new_tuple, compare_output_s2,  etpw, compareTuples_s1_comp);
			add_new_tuple_device<<<1, 1, 0, cudastream3>>>(new_tuple, stream2, etpw, compareTuples_s2_comp);

			cudaStreamSynchronize(cudastream1);
			//cudaStreamSynchronize(cudastream3);
			
			// No other kenel is invoced until the result is saved 
			int *compare_output_new = (int *)malloc(((etpw / 32) + 1)* sizeof(int));
			std::memcpy(compare_output_new, compare_output_s2, sizeof(int) * ((etpw / 32) + 1));
			//queue.push(queueElm(compare_output_new, stream2, new_tuple));
			pool.enqueue([new_tuple, compare_output_new] (){ write_result(interprete(new_tuple, compare_output_new, stream2));});

			j++;
			if (((i+j) % 10000) == 0) printf("%d\n", i+j);
		}
    }
	//write_result(interprete(tuple_prev, compare_output_prev, stream_prev));
	END_M(_)	
}



int main(int argc, char *argv[]){
	if (argc != 5){
		printf("Usage: hellsjoin_file [filename1] [filename2] [rows] [window]");
	}
	etpw = atoi(argv[4]);
	blocksize = 32;	           // Number of threads per block
	gridsize = (etpw / 1024) + 1;  // Number of blocks
	std::cout << "Blocksize: " << blocksize << " Gridsize: " << gridsize << "\n";
	//int minGridSize;
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blocksize, compare_kernel_ipt, 0, (int)etpw / 32);
	//gridsize = (etpw  + blocksize - 1) / blocksize; 

#if MANAGED
    CUDA_SAFE(cudaMallocManaged((void **)&compareTuples_s1_comp, sizeof(tuple) * etpw));
    CUDA_SAFE(cudaMallocManaged((void **)&compareTuples_s2_comp, sizeof(tuple) * etpw));
    compareTuples_s1_inter = (tuple *)calloc(etpw, sizeof(tuple));
    compareTuples_s2_inter = (tuple *)calloc(etpw, sizeof(tuple));

    CUDA_SAFE(cudaMallocManaged((void **)&compare_output_s1, sizeof(int) * ((etpw / 32) + 1)));
    CUDA_SAFE(cudaMallocManaged((void **)&compare_output_s2, sizeof(int) * ((etpw / 32) + 1)));
    CUDA_SAFE(cudaMallocManaged((void **)&compare_output_prev, sizeof(int) * ((etpw / 32) + 1)));
#elif ZERO_COPY_R
    compareTuples_s1_d = (tuple *)calloc(etpw, sizeof(tuple));
    CUDA_SAFE(cudaHostRegister(compareTuples_s1_d, sizeof(tuple) * etpw, 2));
    compareTuples_s2_d = (tuple *)calloc(etpw, sizeof(tuple));
    CUDA_SAFE(cudaHostRegister(compareTuples_s2_d, sizeof(tuple) * etpw, 2));

    compare_output = (int *)calloc(((etpw / 32) + 1), sizeof(int));
    CUDA_SAFE(cudaHostRegister(compare_output, sizeof(int) * ((etpw / 32) + 1), 0));
#else  // ZERO_COPY_M
    CUDA_SAFE(cudaMalloc((void **)&compareTuples_s1_comp, sizeof(tuple) * etpw));
    CUDA_SAFE(cudaMalloc((void **)&compareTuples_s2_comp, sizeof(tuple) * etpw));
	
    //CUDA_SAFE(cudaHostAlloc((void **)&compareTuples_s1_comp, sizeof(tuple) * etpw,0));
    //CUDA_SAFE(cudaHostAlloc((void **)&compareTuples_s2_comp, sizeof(tuple) * etpw,0));
    compareTuples_s1_inter = (tuple *)calloc(etpw, sizeof(tuple));
    compareTuples_s2_inter = (tuple *)calloc(etpw, sizeof(tuple));
    
	//CUDA_SAFE(cudaMalloc((void **)&compare_output_s1, sizeof(int) * ((etpw / 32) + 1)));
    //CUDA_SAFE(cudaMalloc((void **)&compare_output_s2, sizeof(int) * ((etpw / 32) + 1)));

    CUDA_SAFE(cudaHostAlloc((void **)&compare_output_s1, sizeof(int) * ((etpw / 32) + 1), 0));
    CUDA_SAFE(cudaHostAlloc((void **)&compare_output_s2, sizeof(int) * ((etpw / 32) + 1), 0));
    CUDA_SAFE(cudaHostAlloc((void **)&compare_output_prev, sizeof(int) * ((etpw / 32) + 1), 0));
#endif

    myfile.open ("result.csv");
#if FILE_
	startFileTest(argv[1],argv[2],atoi(argv[3])); // Filename1, Filename2, rows
#else
	startManualTest();    
#endif
    myfile.close();
}
