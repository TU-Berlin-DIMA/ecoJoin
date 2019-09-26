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
void compare_kernel_ipt(tuple *input, int count, int *output, size_t etpw, tuple *compareTuples) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	long  global_threads = blockDim.x * gridDim.x;
	
	// Off by one error?
	for (long tuple_id = idx; tuple_id +global_threads< etpw; tuple_id += global_threads){
		int z = 0;
		for (int j = 0; j < count; j++){ // work on batch
			if ((tuple_id+1) * 32 < etpw){
#pragma unroll
				for (int i = 0; i < 32; i++) {
					if  (input[j].key == compareTuples[tuple_id*32+i].key) {
						//printf("%d %d \n", input[j].key, input[j].value);
						z = z | 1 << i;
					}
				}
			} else if (idx * 32 < etpw){
				for (int i = 0; i < etpw - tuple_id*32 ; i++) {
					if  (input[j].key == compareTuples[tuple_id*32+i].key) {
						//printf("%d %d \n", input[j].key, input[j].value);
						z = z | 1 << i;
					}
				}
			}
			output[j*((etpw / 32) + 1)+tuple_id] = z;
		}
	}
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
		//printf("add_new_tuple_device %d %d at %d \n", new_tuple.timestamp, new_tuple.key, currentFIFO_s1);
		if(++currentFIFO_s1 == etpw)
			currentFIFO_s1 = 0;
	} else {
		compareTuples[currentFIFO_s2].key = new_tuple.key;
		compareTuples[currentFIFO_s2].value = new_tuple.value;
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
			printf("(%d, %d) ", compareTuples[i].key);
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
		//compareTuples_s1_inter[currentFIFO_s1_inter].timestamp = input.timestamp;
		currentFIFO_s1_inter++;
		if(currentFIFO_s1_inter == etpw)
			currentFIFO_s1_inter = 0;
	} else {
		compareTuples_s2_inter[currentFIFO_s2_inter].key = input.key;
		compareTuples_s2_inter[currentFIFO_s2_inter].value = input.value;
		//compareTuples_s2_inter[currentFIFO_s2_inter].timestamp = input.timestamp;
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
						printf("%d\n",compareTuples_s2_inter[j].key);	
						DEBUG_P( "Match  ")
						printRecord(input);
						printRecord(compareTuples_s2_inter[j]);
						
						r.key = compareTuples_s2_inter[j].key;
						r.left_value = compareTuples_s2_inter[j].value;
						r.right_value = input.value;
						//r.timestamp = compareTuples_s2_inter[j].timestamp;
					} else { 
						printf("%d\n",compareTuples_s2_inter[j].key);	
						DEBUG_P("Match  ")
						printRecord(input);
						printRecord(compareTuples_s1_inter[j]);

						r.key = compareTuples_s1_inter[j].key;
						r.left_value = compareTuples_s1_inter[j].value;
						r.right_value = input.value;
						//r.timestamp = compareTuples_s1_inter[j].timestamp;
					}
					result.push_back(r);
				}
			}
			bitmap[i / 32] = 0;
		}
    }
	return result;
}

void print_result(std::vector<record2> result) {
    for (auto a : result)
		std::cout << "match  newtuple ("  << a.key << ", "
		<< a.right_value << ", " << a.left_value << ") \n";
}

void write_result(std::vector<record2> result){
    for (auto a : result)
		myfile  << "match  newtuple (" << a.key << ", "
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

		tup[row] = {std::stoi(key), std::stoi(val)};
		row++;
	}
	std::cout << filename << ": " << row << "rows loaded" << std::endl;
};


void startFileTest(std::string filename1, std::string filename2, int rows){
	tuple *records0;
	tuple *records1;
	CUDA_SAFE(cudaHostAlloc((void **)&records0, sizeof(tuple) * rows, 0));
	CUDA_SAFE(cudaHostAlloc((void **)&records1, sizeof(tuple) * rows, 0));


	int n = rows;
	std::vector<int> tmp1, tmp2;
    tmp1.reserve(n);
    tmp2.reserve(n);
    double sel = 0.2;
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
	END_M(_)	
}

int main(int argc, char *argv[]){

	if (argc != 6){
		printf("Usage: hellsjoin_file [filename1] [filename2] [rows] [window] [batchsize]");
	}
	etpw = atoi(argv[4]);
	batchsize = atoi(argv[5]);

    CUDA_SAFE(cudaHostAlloc((void **)&compareTuples_s1_comp, sizeof(tuple) * etpw, 0));
    CUDA_SAFE(cudaHostAlloc((void **)&compareTuples_s2_comp, sizeof(tuple) * etpw, 0));
    compareTuples_s1_inter = (tuple *)calloc(etpw, sizeof(tuple));
    compareTuples_s2_inter = (tuple *)calloc(etpw, sizeof(tuple));
 
    CUDA_SAFE(cudaHostAlloc((void **)&compare_output_s1, sizeof(int) * ((etpw / 32) + 1) * (1+batchsize), 0));
    CUDA_SAFE(cudaHostAlloc((void **)&compare_output_s2, sizeof(int) * ((etpw / 32) + 1) * (1+batchsize), 0));
    CUDA_SAFE(cudaHostAlloc((void **)&compare_output_prev, sizeof(int) * ((etpw / 32) + 1) *(1+batchsize), 0));


    myfile.open ("result.csv");
#if FILE_
	startFileTest(argv[1],argv[2],atoi(argv[3])); // Filename1, Filename2, rows
#else
	startManualTest();    
#endif
    myfile.close();
}
