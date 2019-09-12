#include <bitset>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mutex>
#include <vector>
#include <chrono>
#include <thread>
#include <cstring>

#include "hashtbl.h"

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

void printRecord(tuple rec){
	DEBUG_P("key: " << rec.key << " timestamp:  "<< rec.timestamp << " value: " << rec.value)
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
					int z = (j / 32) + (( j % 32) * 32);
					record2 r;
					if (stream == stream1) {
						DEBUG_P( "Match  ")
						printRecord(input);
						printRecord(compareTuples_s2_inter[z]);
						
						r.key = compareTuples_s2_inter[z].key;
						r.left_value = compareTuples_s2_inter[z].value;
						r.right_value = input.value;
						r.timestamp = compareTuples_s2_inter[z].timestamp;
					} else { 
						DEBUG_P("Match  ")
						printRecord(input);
						printRecord(compareTuples_s1_inter[z]);

						r.key = compareTuples_s1_inter[z].key;
						r.left_value = compareTuples_s1_inter[z].value;
						r.right_value = input.value;
						r.timestamp = compareTuples_s1_inter[z].timestamp;
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


void startManualTest(){
	// Key, timestamp, value
    tuple records0[10]{{1, 1, 11}, 
		{2, 1, 12}, 
		{3, 1, 13}, 
		{4, 1, 14},
		{5, 1, 15},
		{2, 2, 16},
		{6, 3, 17}, 
		{6, 3, 18}, 
		{7, 3, 19}, 
		{8, 3, 110}};
    tuple records1[14]{{1, 1, 21},  
		{2, 1, 22},  
		{3, 1, 23},  
		{4, 1, 24},  
		{5, 1, 25},
		{1,	2, 26},  
		{2, 2, 27}, 
	   	{2, 2, 28},  
		{4, 2, 29},
		{1, 3, 210}, 
		{3, 3, 211},
	   	{6, 3, 212},
	   	{4, 3, 213},
	   	{6, 3, 214}};

	HashtblConfig config;
	config.table_size = 300;
	config.cuda_blocksize = 0;
	config.cuda_numblocks = 0;
	Hashtbl tbl = Hashtbl(config);
   
	unsigned output_size = 0;
	unsigned *output = (unsigned *) calloc(config.table_size  *2, sizeof(unsigned));
	
	unsigned *vals_1 = (unsigned *) malloc(sizeof(unsigned) * 10);
	unsigned *keys_1 = (unsigned *) malloc(sizeof(unsigned) * 10);

	unsigned *vals_2 = (unsigned *) malloc(sizeof(unsigned) * 14);
	unsigned *keys_2 = (unsigned *) malloc(sizeof(unsigned) * 14);

	for (int i = 0; i < 10; i++) {
		vals_1[i] = (unsigned) records0[i].value;
		keys_1[i] = (unsigned) records0[i].key;
	}
	
	for (int i = 0; i < 14; i++) {
		vals_2[i] = (unsigned) records1[i].value;
		keys_2[i] = (unsigned) records1[i].key;
	}

	// Building and probing is done on the same HT
	START_M(_)	
	printf("build\n");
	tbl.Build(10, keys_1, vals_1);
	cudaDeviceSynchronize();
	//printf("probe\n");
	//tbl.Probe(10, keys_1, vals_1, &output_size, output);

	//for (int i = 0; i < output_size; i++)
	//	std::cout << output[i] << " ,";
	//std::cout << std::endl;

	//printf("build\n");
	//tbl.Build(14, keys_2, vals_2);
	printf("probe\n");
	tbl.Probe(14, keys_2, vals_2, &output_size, output);
	
	for (int i = 0; i < output_size; i++)
		std::cout << output[i] << " ,";
	std::cout << std::endl;
	END_M(_)
}

int main(int argc, char *argv[]){

	/*if (argc != 2){
		printf("Usage: hellsjoin_file [window]");
	}
	etpw = atoi(argv[1]);
	blocksize = 32;	           // Number of threads per block
	gridsize = (etpw / 1024) + 1;  // Number of blocks
	std::cout << "Blocksize: " << blocksize << " Gridsize: " << gridsize << "\n";
	*/
	startManualTest();    
}
