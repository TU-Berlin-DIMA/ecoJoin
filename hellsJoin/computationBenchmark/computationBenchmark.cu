#include <bitset>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mutex>
#include <vector>
#include <chrono>
#include <thread>
#include <cstring>
#include <cstdio>
#include "hashtbl.h"

/***************************************************************************************/
/***************************************************************************************/

#define etpw 1000

/***************************************************************************************/
/***************************************************************************************/

struct tuple {
    int key;
    int timestamp;
    int value;
};

int *compare_output_h, *compare_output_g;
tuple *compareTuples_h, *compareTuples_g, *new_tuples_h, *new_tuples_g;
int blocksize, gridsize;


/***************************************************************************************/
/***************************************************************************************/

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
	long elapsed_seconds_name = std::chrono::duration_cast<std::chrono::nanoseconds> (end_name-start_name).count(); \
    std::cout << elapsed_seconds_name << "\n"; 
    //std::cout << name << " " << elapsed_seconds_name << " ns\n"; 

void write_result(int *compareResult, int count, std::string filename){
	std::ofstream myfile;
	myfile.open(filename, std::ios_base::app);
    for (int i = 0; i < count; i++){
		myfile << compareResult[i] << '\n';
	}
	myfile.close();
}

void interpreteAndWrite(tuple input, int *bitmap, std::string filename, tuple *compareTuples) { 
	std::ofstream myfile;
	myfile.open(filename, std::ios_base::app);
	for (int i = 0; i < etpw; i = i + 32) {
		if (bitmap[i / 32] == 0) { // first check
			continue;
		} else {
			for (int k = 0; k < 32; k++){
				int j = i+k;
				if (std::bitset<32>(bitmap[j / 32]).test(j % 32)) { // fine-grained check
					int z = j;//(j / 32) + (( j % 32) * 32);
					myfile  << compareTuples[z].key << " "
							<< compareTuples[z].value << " "
							<< input.value << " "
							<< compareTuples[z].timestamp << "\n";
				}
			}
			bitmap[i / 32] = 0;
		}
	}
	myfile.close();
}

/*void interpreteAndWriteBatch(tuple *input, int tupel_Count,int *bitmap, std::string filename) { 
	std::ofstream myfile;
	myfile.open(filename, std::ios_base::app);
	for (int h = 0; h < tuple_Count; h++) {
		for (int i = 0; i < etpw; i = i + 32) {
			if (bitmap[(i+ h*((etpw/32)+1)) / 32] == 0) { // first check
				continue;
			} else {
				for (int k = 0; k < 32; k++){
					int j = i+k;
					if (std::bitset<32>(bitmap[(i+ h*((etpw/32)+1)) / 32]).test(j % 32)) { // fine-grained check
						int z = j;//(j / 32) + (( j % 32) * 32);
						myfile << compareTuples_s2_inter[z].key << " "
							   << compareTuples_s2_inter[z].value << " "
							   << input[h].value << " "
							   << compareTuples_s2_inter[z].timestamp << "\n";
					}
				}
				bitmap[(i+ h*((etpw/32)+1)) / 32] = 0;
			}
		}
	}
	myfile.close();
}*/

/***************************************************************************************/
/***************************************************************************************/

__global__
void compare_kernel_ipt(tuple input, int *output, tuple *compareTuples) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    int z = 0;
    if ((idx+1) * 32 < etpw){
#pragma unroll
		for (int i = 0; i < 32; i++) {
			if  ((input.timestamp == compareTuples[idx*32+i].timestamp)  // Time Window
					&& (input.key == compareTuples[idx*32+i].key)) {
      //        printf("%d %d \n", input.key, input.timestamp);
				z = z | 1 << i;
			}
		}
		output[idx] = z;
    } else if (idx * 32 < etpw){
        for (int i = 0; i < etpw - idx*32 ; i++) {
            if  ((input.timestamp == compareTuples[idx*32+i].timestamp)  // Time Window
                    && (input.key == compareTuples[idx*32+i].key)) {
    //          printf("%d %d \n", input.key, input.timestamp);
                z = z | 1 << i;
            }
        }
		output[idx] = z;
    }
}

// Computes (Count * Integer) per thread
__global__
void compare_kernel_ipt_batch(tuple *input, int count, int *output, tuple *compareTuples) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    int z = 0;
	for (int j = 0; j < count && (idx + j * ((etpw / 32) +1)) < (count * ((etpw / 32) +1)); j++){
		tuple t = input[j];
        if ((idx+1) * 32 < etpw){
#pragma unroll
            for (int i = 0; i < 32; i++) {
                if  ((t.timestamp == compareTuples[idx*32+i].timestamp)  // Time Window
                        && (t.key == compareTuples[idx*32+i].key)) {
                    printf("%d %d \n", input[j].key, input[j].timestamp);
                    z = z | 1 << i;
                }
            }
        } else if (idx * 32 < etpw){
            for (int i = 0; i < etpw - idx*32 ; i++) {
                if  ((t.timestamp == compareTuples[idx*32+i].timestamp)  // Time Window
                        && (t.key == compareTuples[idx*32+i].key)) {
                    printf("%d %d \n", input[j].key, input[j].timestamp);
                    z = z | 1 << i;
                }
            }
        }
        output[idx + j * ((etpw / 32) +1)] = z;
    }
}

__global__
void compare_kernel_ipt_batch_shared(tuple *input, int count, int *output, tuple *compareTuples) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    int z = 0;
	__shared__ tuple s;

    for (int j = 0; j < count; j++){
        if ((idx+1) * 32 < etpw){
			//s[j] = input[j];
#pragma unroll
            for (int i = 0; i < 32; i++) {
                if  ((input[j].timestamp == compareTuples[idx*32+i].timestamp)  // Time Window
                        && (input[j].key == compareTuples[idx*32+i].key)) {
                    //printf("%d %d \n", input[j].key, input[j].timestamp);
                    z = z | 1 << i;
                }
            }
        } else if (idx * 32 < etpw){
            for (int i = 0; i < etpw - idx*32 ; i++) {
                if  ((input[j].timestamp == compareTuples[idx*32+i].timestamp)  // Time Window
                        && (input[j].key == compareTuples[idx*32+i].key)) {
                    //printf("%d %d \n", input[j].key, input[j].timestamp);
                    z = z | 1 << i;
                }
            }
        }
        output[j*((etpw / 32) + 1)+idx] = z;
    }
}

/***************************************************************************************/
/***************************************************************************************/

void nestedLoopCompare_cCPU_mCPU(tuple *new_tuples, int new_tuplesCount, int *compare_output, tuple *compareTuples){
	START_M("nestedLoopCompare_cCPU_mCPU")	
	int z = 0;
	for (int i = 0; i < new_tuplesCount; i++) {
#pragma unroll
		for (int j = 0; j < etpw; j++) {
			if  ((new_tuples[i].timestamp == compareTuples[j].timestamp)  // Time Window
						&& (new_tuples[i].key == compareTuples[j].key)) {
	            //printf("%d %d %d %d\n", new_tuples[i].key, new_tuples[i].timestamp, compareTuples[j].key, compareTuples[j].timestamp);
				compare_output[j / 32] = compare_output[j / 32] | 1 << (j % 32);
			}
		}
#ifdef DEBUG
		write_result(compare_output, ((etpw/32)+1), std::string("comp_results/nestedLoopCompare_cCPU_mCPU.txt"));
		//interpreteAndWrite(new_tuples[i], compare_output, std::string("comp_results/nestedLoopCompare_cCPU_mCPU.txt"), compareTuples);
#endif
	}
    END_M("nestedLoopCompare_cCPU_mCPU")	
}

void nestedLoopCompare_cGPU_mCPU(tuple *new_tuples, int new_tuplesCount, int *compare_output, tuple *compareTuples){
	CUDA_SAFE(cudaHostAlloc((void **)&compare_output_h, sizeof(int) * ((etpw / 32) + 1), 0));
    CUDA_SAFE(cudaHostAlloc((void **)&compareTuples_h, sizeof(tuple) * etpw, 0));
	std::memcpy(compareTuples_h, compareTuples, sizeof(tuple) * etpw);

    START_M("nestedLoopCompare_cGPU_mCPU")	
	for (int i = 0; i < new_tuplesCount; i++) {
		compare_kernel_ipt<<<blocksize, gridsize>>>(new_tuples[i], compare_output_h, compareTuples_h);
		CUDA_SAFE(cudaStreamSynchronize(0));
#ifdef DEBUG
		write_result(compare_output, etpw, std::string("comp_results/nestedLoopCompare_cGPU_mCPU.txt"));
#endif
	}
	compare_output = compare_output_h;
    END_M("nestedLoopCompare_cGPU_mCPU")	
	CUDA_SAFE(cudaFreeHost(compare_output_h));
	CUDA_SAFE(cudaFreeHost(compareTuples_h));
}

void nestedLoopCompare_cGPU_mGPU(tuple *new_tuples, int new_tuplesCount, int *compare_output, tuple *compareTuples){
	CUDA_SAFE(cudaMalloc((void **)&compare_output_g, sizeof(int) * ((etpw / 32) + 1)));
    CUDA_SAFE(cudaMalloc((void **)&compareTuples_g, sizeof(tuple) * etpw));
	
    START_M("nestedLoopCompare_cGPU_mGPU")	
	CUDA_SAFE(cudaMemcpy(compareTuples_g, compareTuples, sizeof(tuple) * etpw, cudaMemcpyHostToDevice));
	for (int i = 0; i < new_tuplesCount; i++) {
		compare_kernel_ipt<<<blocksize, gridsize>>>(new_tuples[i], compare_output_g, compareTuples_g);
		CUDA_SAFE(cudaStreamSynchronize(0));
#ifdef DEBUG
		CUDA_SAFE(cudaMemcpy(compare_output, compare_output_g, sizeof(int) * ((etpw / 32) + 1), cudaMemcpyDeviceToHost));
		write_result(compare_output, etpw, std::string("comp_results/nestedLoopCompare_cGPU_mGPU.txt"));
#endif
	}
	CUDA_SAFE(cudaMemcpy(compare_output, compare_output_g, sizeof(int) * ((etpw / 32) + 1), cudaMemcpyDeviceToHost));
    END_M("nestedLoopCompare_cGPU_mGPU")
	
	CUDA_SAFE(cudaFree(compare_output_g));
	CUDA_SAFE(cudaFree(compareTuples_g));
}

void nestedLoopCompareBatch_cGPU_mCPU(tuple *new_tuples, int new_tuplesCount, int *compare_output, tuple *compareTuples){
	CUDA_SAFE(cudaHostAlloc((void **)&compare_output_h, sizeof(int) * ((etpw / 32) + 1) * new_tuplesCount, 0));
    CUDA_SAFE(cudaHostAlloc((void **)&compareTuples_h, sizeof(tuple) * etpw, 0));
    CUDA_SAFE(cudaHostAlloc((void **)&new_tuples_h, sizeof(tuple) * new_tuplesCount, 0));
	std::memcpy(compareTuples_h, compareTuples, sizeof(tuple) * etpw);
	std::memcpy(new_tuples_h, new_tuples, sizeof(tuple) * new_tuplesCount);

    START_M("nestedLoopCompareBatch_cGPU_mCPU")	
	compare_kernel_ipt_batch<<<blocksize, gridsize>>>(new_tuples_h, new_tuplesCount, compare_output_h, compareTuples_h);
	CUDA_SAFE(cudaStreamSynchronize(0));
	compare_output = compare_output_h;
    END_M("nestedLoopCompareBatch_cGPU_mCPU")	

	CUDA_SAFE(cudaFreeHost(compare_output_h));
	CUDA_SAFE(cudaFreeHost(compareTuples_h));
	CUDA_SAFE(cudaFreeHost(new_tuples_h));
}

void nestedLoopCompareBatch_cGPU_mGPU(tuple *new_tuples, int new_tuplesCount, int *compare_output, tuple *compareTuples){
	CUDA_SAFE(cudaMalloc((void **)&compare_output_g, sizeof(int) * ((etpw / 32) + 1) * new_tuplesCount));
    CUDA_SAFE(cudaMalloc((void **)&compareTuples_g, sizeof(tuple) * etpw));
    CUDA_SAFE(cudaMalloc((void **)&new_tuples_g, sizeof(tuple) * new_tuplesCount));
	compare_output  = (int*) calloc(((etpw/32)+1) * new_tuplesCount, sizeof(int));
	
    START_M("nestedLoopCompareBatch_cGPU_mGPU")	
	CUDA_SAFE(cudaMemcpy(compareTuples_g, compareTuples, sizeof(tuple) * etpw, cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(new_tuples_g, new_tuples, sizeof(tuple) * new_tuplesCount, cudaMemcpyHostToDevice));
	compare_kernel_ipt_batch<<<blocksize, gridsize>>>(new_tuples_g, new_tuplesCount, compare_output_g, compareTuples_g);
	CUDA_SAFE(cudaMemcpy(compare_output, compare_output_g, sizeof(int) * ((etpw / 32) + 1) * new_tuplesCount, cudaMemcpyDeviceToHost));
#ifdef DEBUG
	write_result(compare_output, ((etpw/32)+1) * new_tuplesCount, std::string("comp_results/nestedLoopCompareBatch_cGPU_mGPU.txt"));
#endif
    END_M("nestedLoopCompareBatch_cGPU_mGPU")	
	CUDA_SAFE(cudaFree(compare_output_g));
	CUDA_SAFE(cudaFree(compareTuples_g));
	CUDA_SAFE(cudaFree(new_tuples_g));
}

void nestedLoopCompareBatchOnlyComp_cGPU_mGPU(tuple *new_tuples, int new_tuplesCount, int *compare_output, tuple *compareTuples){
	CUDA_SAFE(cudaMalloc((void **)&compare_output_g, sizeof(int) * ((etpw / 32) + 1) * new_tuplesCount));
    CUDA_SAFE(cudaMalloc((void **)&compareTuples_g, sizeof(tuple) * etpw));
    CUDA_SAFE(cudaMalloc((void **)&new_tuples_g, sizeof(tuple) * new_tuplesCount));
	compare_output  = (int*) calloc(((etpw/32)+1) * new_tuplesCount, sizeof(int));
	
	CUDA_SAFE(cudaMemcpy(compareTuples_g, compareTuples, sizeof(tuple) * etpw, cudaMemcpyHostToDevice));
	CUDA_SAFE(cudaMemcpy(new_tuples_g, new_tuples, sizeof(tuple) * new_tuplesCount, cudaMemcpyHostToDevice));

    START_M("nestedLoopCompareBatchCompOnly_cGPU_mGPU")	
	compare_kernel_ipt_batch<<<blocksize, gridsize>>>(new_tuples_g, new_tuplesCount, compare_output_g, compareTuples_g);
	CUDA_SAFE(cudaDeviceSynchronize());
    END_M("nestedLoopCompareBatchCompOnly_cGPU_mGPU")	

	CUDA_SAFE(cudaMemcpy(compare_output, compare_output_g, sizeof(int) * ((etpw / 32) + 1) * new_tuplesCount, cudaMemcpyDeviceToHost));
#ifdef DEBUG
	write_result(compare_output, ((etpw/32)+1) * new_tuplesCount, std::string("comp_results/nestedLoopCompareBatchCompOnly_cGPU_mGPU.txt"));
#endif

	free(compare_output);
	CUDA_SAFE(cudaFree(compare_output_g));
	CUDA_SAFE(cudaFree(compareTuples_g));
	CUDA_SAFE(cudaFree(new_tuples_g));
}

void hashJoin_cGPU_mGPU(tuple *new_tuples, int new_tuplesCount, int *compare_output, tuple *compareTuples){
	unsigned etpw_ = etpw;
	HashtblConfig config;
	config.table_size = etpw_ * 2;
	Hashtbl tbl = Hashtbl(config);

	unsigned *compare_out = (unsigned *) calloc(sizeof(unsigned), config.table_size * 3);

	unsigned *vals1 = (unsigned *) malloc(sizeof(unsigned) * new_tuplesCount);
	unsigned *keys1 = (unsigned *) malloc(sizeof(unsigned) * new_tuplesCount);
	unsigned *vals2 = (unsigned *) malloc(sizeof(unsigned) * etpw_);
	unsigned *keys2 = (unsigned *) malloc(sizeof(unsigned) * etpw_);
	
	for (int i = 0; i < new_tuplesCount; i++) {
	     vals1[i] = (unsigned) new_tuples[i].value;
		 keys1[i] = (unsigned) new_tuples[i].key;
	}
	for (int i = 0; i < etpw_; i++) {
	     vals2[i] = (unsigned) compareTuples[i].value;
		 keys2[i] = (unsigned) compareTuples[i].key;
	}

	/* Probe should be smaller
	if (new_tuplesCount < etpw_){
#ifdef DEBUG
		printf("Changing Probe with Build Cols\n");
#endif
		unsigned *tmp = vals1;
		vals1 = vals2;
		vals2 = tmp;
		tmp = keys1;
		keys1 = keys2;
		keys2 = tmp;
		unsigned tm = etpw_;
		etpw_ = new_tuplesCount;
		new_tuplesCount = tm;
	}*/

#ifdef DEBUG
	std::cout << "Probesize " << new_tuplesCount << "\n";
	std::cout << "Buildsize " << etpw_<< "\n";
	std::cout << "Hashtablesize " << config.table_size << "elm " << config.table_size * 2 * sizeof(unsigned) / 1024 / 1024 << "MB \n";
	std::cout << "Estimated Outputsize" << config.table_size * 3 << "elm " << config.table_size * 2 * sizeof(unsigned) * 3 / 1024 / 1024 << "MB \n";
#endif

	unsigned *current_d, *output_d, *keys1_d, *vals1_d, *keys2_d, *vals2_d;
    CUDA_SAFE(cudaMalloc(&keys1_d, sizeof(unsigned) * new_tuplesCount));
    CUDA_SAFE(cudaMalloc(&vals1_d, sizeof(unsigned) * new_tuplesCount));
	CUDA_SAFE(cudaMalloc(&keys2_d, sizeof(unsigned) * etpw_));
	CUDA_SAFE(cudaMalloc(&vals2_d, sizeof(unsigned) * etpw_));
    CUDA_SAFE(cudaMalloc(&output_d, sizeof(unsigned) * config.table_size *3)); 
    CUDA_SAFE(cudaMalloc(&current_d, sizeof(unsigned)));

	int blockSize1, minGridSize1, gridSize1;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize1, &blockSize1, (void *)probe, 0, new_tuplesCount);
	//blockSize1 = 512;
	gridSize1 = ((long) new_tuplesCount+ blockSize1 - 1) / blockSize1;
		
	int blockSize2, minGridSize2, gridSize2;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize2, &blockSize2, (void *)build, 0, etpw_);
	//blockSize2 = 512;
	gridSize2 = ((long) etpw_ + blockSize2 - 1) / blockSize2;
	
#ifdef DEBUG
	std::cout << "G&B probe " << gridSize1 << " " << blockSize1 << "\n";
	std::cout << "G&B build " << gridSize2 << " " << blockSize2 << "\n";
#endif

	START_M("hashJoin")	
	CUDA_SAFE(cudaMemcpy(keys1_d, keys1, sizeof(unsigned) * new_tuplesCount, cudaMemcpyDefault));
    CUDA_SAFE(cudaMemcpy(vals1_d, vals1, sizeof(unsigned) * new_tuplesCount, cudaMemcpyDefault));
    CUDA_SAFE(cudaMemcpy(keys2_d, keys2, sizeof(unsigned) * etpw_, cudaMemcpyDefault));
    CUDA_SAFE(cudaMemcpy(vals2_d, vals2, sizeof(unsigned) * etpw_, cudaMemcpyDefault));

	build<<<gridSize2, blockSize2>>>(etpw_, tbl.content, config.table_size, keys2_d, vals2_d);
	CUDA_SAFE(cudaDeviceSynchronize());
	probe<<<gridSize1, blockSize1>>>(new_tuplesCount, keys1_d, vals1_d, tbl.content, config.table_size, output_d, current_d);

	CUDA_SAFE(cudaMemcpy(compare_out, output_d, config.table_size * sizeof(unsigned) * 3, cudaMemcpyDefault));
    END_M("hashJoin")	

//#ifdef DEBUG
	for(int i = 0; i < config.table_size * 3; i++){
		if (compare_out[i] != 0)
			std::cout << compare_out[i] << ",";
	}
	std::cout << "\n";
//#endif

	CUDA_SAFE(cudaFree(keys1_d));
    CUDA_SAFE(cudaFree(vals1_d));
    CUDA_SAFE(cudaFree(keys2_d));
    CUDA_SAFE(cudaFree(vals2_d));
    CUDA_SAFE(cudaFree(current_d));
    CUDA_SAFE(cudaFree(output_d));
}

void hashJoin_sm_cGPU_mGPU(tuple *new_tuples, int new_tuplesCount, int *compare_output, tuple *compareTuples){
	unsigned etpw_ = etpw;
	HashtblConfig config;
	config.table_size = etpw_ * 2;
	Hashtbl tbl = Hashtbl(config);

	unsigned *compare_out = (unsigned *) calloc(sizeof(unsigned), config.table_size * 3);

	unsigned *vals1 = (unsigned *) malloc(sizeof(unsigned) * new_tuplesCount);
	unsigned *keys1 = (unsigned *) malloc(sizeof(unsigned) * new_tuplesCount);
	unsigned *vals2 = (unsigned *) malloc(sizeof(unsigned) * etpw_);
	unsigned *keys2 = (unsigned *) malloc(sizeof(unsigned) * etpw_);
	
	for (int i = 0; i < new_tuplesCount; i++) {
	     vals1[i] = (unsigned) new_tuples[i].value;
		 keys1[i] = (unsigned) new_tuples[i].key;
	}
	for (int i = 0; i < etpw_; i++) {
	     vals2[i] = (unsigned) compareTuples[i].value;
		 keys2[i] = (unsigned) compareTuples[i].key;
	}

	/* Probe should be smaller
	if (new_tuplesCount < etpw_){
#ifdef DEBUG
		printf("Changing Probe with Build Cols\n");
#endif
		unsigned *tmp = vals1;
		vals1 = vals2;
		vals2 = tmp;
		tmp = keys1;
		keys1 = keys2;
		keys2 = tmp;
		unsigned tm = etpw_;
		etpw_ = new_tuplesCount;
		new_tuplesCount = tm;
	}*/

#ifdef DEBUG
	std::cout << "Probesize " << new_tuplesCount << "\n";
	std::cout << "Buildsize " << etpw_<< "\n";
	std::cout << "Hashtablesize " << config.table_size << "elm " << config.table_size * 2 * sizeof(unsigned) / 1024 / 1024 << "MB \n";
	std::cout << "Estimated Outputsize" << config.table_size * 3 << "elm " << config.table_size * 2 * sizeof(unsigned) * 3 / 1024 / 1024 << "MB \n";
#endif

	unsigned *current_d, *output_d, *keys1_d, *vals1_d, *keys2_d, *vals2_d;
    CUDA_SAFE(cudaMalloc(&keys1_d, sizeof(unsigned) * new_tuplesCount));
    CUDA_SAFE(cudaMalloc(&vals1_d, sizeof(unsigned) * new_tuplesCount));
	CUDA_SAFE(cudaMalloc(&keys2_d, sizeof(unsigned) * etpw_));
	CUDA_SAFE(cudaMalloc(&vals2_d, sizeof(unsigned) * etpw_));
    CUDA_SAFE(cudaMalloc(&output_d, sizeof(unsigned) * config.table_size *3)); 
    CUDA_SAFE(cudaMalloc(&current_d, sizeof(unsigned)));

	int blockSize1, minGridSize1, gridSize1;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize1, &blockSize1, (void *)probe, 0, new_tuplesCount);
	//blockSize1 = 512;
	gridSize1 = ((long) new_tuplesCount+ blockSize1 - 1) / blockSize1;
		
	int blockSize2, minGridSize2, gridSize2;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize2, &blockSize2, (void *)build, 0, etpw_);
	//blockSize2 = 512;
	gridSize2 = ((long) etpw_ + blockSize2 - 1) / blockSize2;
	
#ifdef DEBUG
	std::cout << "G&B probe " << gridSize1 << " " << blockSize1 << "\n";
	std::cout << "G&B build " << gridSize2 << " " << blockSize2 << "\n";
#endif

	START_M("hashJoin")	
	CUDA_SAFE(cudaMemcpy(keys1_d, keys1, sizeof(unsigned) * new_tuplesCount, cudaMemcpyDefault));
    CUDA_SAFE(cudaMemcpy(vals1_d, vals1, sizeof(unsigned) * new_tuplesCount, cudaMemcpyDefault));
    CUDA_SAFE(cudaMemcpy(keys2_d, keys2, sizeof(unsigned) * etpw_, cudaMemcpyDefault));
    CUDA_SAFE(cudaMemcpy(vals2_d, vals2, sizeof(unsigned) * etpw_, cudaMemcpyDefault));


	build_and_probe_sm<<<gridSize2, blockSize2, sizeof(unsigned) * etpw  * 4 >>>(config.table_size, etpw_, keys2_d, vals2_d, new_tuplesCount, keys1_d, keys1_d, output_d,  current_d, tbl.content);

	CUDA_SAFE(cudaMemcpy(compare_out, output_d, config.table_size * sizeof(unsigned) * 3, cudaMemcpyDefault));
    END_M("hashJoin")	

//#ifdef DEBUG
	for(int i = 0; i < config.table_size * 3; i++){
		if (compare_out[i] != 0)
			std::cout << compare_out[i] << ",";
	}
	std::cout << "\n";
//#endif

	CUDA_SAFE(cudaFree(keys1_d));
    CUDA_SAFE(cudaFree(vals1_d));
    CUDA_SAFE(cudaFree(keys2_d));
    CUDA_SAFE(cudaFree(vals2_d));
    CUDA_SAFE(cudaFree(current_d));
    CUDA_SAFE(cudaFree(output_d));
}

//TODO.
void nestedLoopCompareBatchShared_cGPU_mCPU(tuple *new_tuples, int new_tuplesCount, int *compare_output, tuple *compareTuples){
	CUDA_SAFE(cudaHostAlloc((void **)&compare_output_h, sizeof(int) * ((etpw / 32) + 1), 0));
    CUDA_SAFE(cudaHostAlloc((void **)&compareTuples_h, sizeof(tuple) * etpw, 0));
	std::memcpy(compareTuples_h, compareTuples, sizeof(tuple) * etpw);

    START_M("nestedLoopCompareBatchShared_cGPU_mCPU")	
	//compare_kernel_ipt_batch_shared<<<blocksize, gridsize>>>(new_tuples, compare_output_h,  compareTuples_h);
	compare_output = compare_output_h;
    END_M("nestedLoopCompareBatchShared_cGPU_mCPU")	
}

/***************************************************************************************/
/***************************************************************************************/

void resetFiles(){
	std::vector<std::string> files = {"comp_results/nestedLoopCompare_cCPU_mCPU.txt",
						"comp_results/nestedLoopCompare_cGPU_mCPU.txt",
						"comp_results/nestedLoopCompare_cGPU_mGPU.txt",
						"comp_results/nestedLoopCompareBatch_cGPU_mCPU.txt",
						"comp_results/nestedLoopCompareBatch_cGPU_mGPU.txt",
						"comp_results/nestedLoopCompareBatchCompOnly_cGPU_mGPU.txt",
						"comp_results/hashJoin.txt",
						"comp_results/nestedLoopCompareBatchShared_cGPU_mCPU.txt"};
	std::ofstream myfile;
	for (auto a : files){
		remove(a.c_str());
	}
}

void parseCSV(std::string filename, tuple *tup, int tuplesCount){
	std::ifstream file(filename);
	std::string line;
	int row = 0;
	while (std::getline(file, line) && row < tuplesCount){
		std::stringstream iss(line);
		std::string key, time, val;
		std::getline(iss, key , ',');
		std::getline(iss, time, ',');
		std::getline(iss, val , ',');

		tup[row] = {std::stoi(key), std::stoi(time), std::stoi(val)};
		row++;
	}
	std::cout << filename << ": " << row << " rows loaded" << std::endl;
};

/***************************************************************************************/
/***************************************************************************************/

void startFileTest(std::string filename1, std::string filename2, int rows){;
	tuple *new_tuples    = new tuple[rows];
	tuple *compareTuples = new tuple[etpw];
	int *compare_output  = (int*) calloc((etpw / 32)+1, sizeof(int));

	parseCSV(filename1.c_str(), new_tuples, rows);
	parseCSV(filename2.c_str(), compareTuples, etpw);

	int runs = 1;


/*	printf("nestedLoopCompare_cCPU_mCPU\n");
	for (int i = 0; i < runs; i++){ 
		nestedLoopCompare_cCPU_mCPU(new_tuples, rows, compare_output, compareTuples);
	}*/
	/*
	printf("nestedLoopCompare_cGPU_mGPU\n");
	for (int i = 0; i < runs; i++){ 
		nestedLoopCompare_cGPU_mGPU(new_tuples, rows, compare_output, compareTuples);
	}

	printf("nestedLoopCompare_cGPU_mCPU\n");
	for (int i = 0; i < runs; i++){ 
		nestedLoopCompare_cGPU_mCPU(new_tuples, rows, compare_output, compareTuples);
	}

	printf("nestedLoopCompareBatch_cGPU_mCPU\n");
	for (int i = 0; i < runs; i++){ 
		nestedLoopCompareBatch_cGPU_mCPU(new_tuples, rows, compare_output, compareTuples);
	}
	printf("nestedLoopCompareBatch_cGPU_mGPU\n");
	for (int i = 0; i < runs; i++){ 
		nestedLoopCompareBatch_cGPU_mGPU(new_tuples, rows, compare_output, compareTuples);
	}
*/
	for (int i = 0; i < runs; i++){ 
		hashJoin_cGPU_mGPU(new_tuples, rows, compare_output, compareTuples);
	}

/*
	for (int i = 0; i < runs; i++){ 
		hashJoin_sm_cGPU_mGPU(new_tuples, rows, compare_output, compareTuples);
	}*/

	/*printf("nestedLoopCompareBatchOnlyComp_cGPU_mGPU\n");
	for (int i = 0; i < runs; i++){
		nestedLoopCompareBatchOnlyComp_cGPU_mGPU(new_tuples, rows, compare_output, compareTuples);
	}*/
	//for (int i = 0; i < runs; i++){ 
	//	nestedLoopCompareBatchShared_cGPU_mCPU(tuple *new_tuples, int compare_output, tuple *compareTuples);
	//}

}

int main(int argc, char *argv[]){
	if (argc != 4){
		printf("Usage: hellsjoin_file [filename1] [filename2] [inputtuples]");
	}

	blocksize = 32;	           // Number of threads per block
	gridsize = (etpw / (blocksize  * 32)) + 1;  // Number of blocks
	std::cout << "Blocksize: " << blocksize << " Gridsize: " << gridsize << "\n";

	resetFiles();
	startFileTest(argv[1],argv[2],atoi(argv[3])); // Filename1, Filename2, rows
}
