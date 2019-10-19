/*
 *  File: hashtbl_um.cu
 *
 *  Inspired by Karnagel paper.
 *
 *  Variables used with Unified Memory:
 *      content
 *      keys, vals
 *      input, output
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "benchmark_helper_lib_v5.h"

#include "MurmurHash.h"
#include "hashtbl.h"

#define MAX_ITERATION_ATTEMPTS 333000000

//__device__ __managed__ int num_groups = 0;
/*******************************************************************************/
/*******************************************************************************/
/* Shared Memory */
__global__
void build_linprobe_shared(const int input_size, const uint64_t table_size, 
		const int *keys, const int *vals) {
	long  global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    long  global_threads = blockDim.x * gridDim.x;
	extern __shared__ int content[];

    long tuple_id = global_idx; 
    //for (long tuple_id = global_idx; 
	//		tuple_id < input_size; 
	//		tuple_id += global_threads ){
		uint64_t location = keys[tuple_id];
		//location *= HASH_FACTOR;
		//location &= table_size;
		//location = location % table_size;
		//location = (13 ^ location ) & table_size;
		location &= (table_size-1);

		for (int j = 0; MAX_ITERATION_ATTEMPTS > j; j++) {
			int current_key = content[location * 2];
			if (current_key == 0) { 
				int old = atomicCAS(&content[location * 2], 0, keys[tuple_id]);
				if (old == 0) {
					content[location * 2 + 1] = vals[tuple_id];
					break;
				}
			}
			if ((++location)*2 == table_size)
				location = 0;   
		}
    //}
}

/*******************************************************************************/
/*******************************************************************************/
/* LSB */

//shared memory ht
// gpu 3

#define HASH_FACTOR 123456789123456789ull
__global__
void build_linprobe_lsb(const int input_size, int *content, 
					 const uint64_t buckets, const int *keys, 
					 const int *vals) {
	long  global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    long  global_threads = blockDim.x * gridDim.x;

    for (long tuple_id = global_idx; tuple_id < input_size; tuple_id += global_threads ){
		uint64_t currentBucket = keys[tuple_id];
		// (1)
		//currentBucket = (13 ^ currentBucket) & (buckets- 1);
		//currentBucket = currentBucket % buckets;
		// (2) 
		//currentBucket = currentBucket << 1;
		//currentBucket *= HASH_FACTOR;
		currentBucket &= buckets-1;

		for (int j = 0; MAX_ITERATION_ATTEMPTS > j; j++) {
			//printf("key: %u, j:%u, loc:%u\n",keys[tuple_id], j, currentBucket );
			int current_key = content[currentBucket * 2];
			if (current_key == 0) { 
				//content[location * 2] = keys[tuple_id];
				int old = atomicCAS(&content[currentBucket * 2], 0, keys[tuple_id]);
				if (old == 0) {
					content[currentBucket * 2 + 1] = vals[tuple_id];
					break;
				}
			} 
			if ((++currentBucket)*2 == buckets)
				 currentBucket = 0;
		}
    }
}

/*
 * Probes input_size keys and values with the hash table.
 * Each Inputtuple one thread
 */
__global__ 
void probe_linprobe_lsb(const int input_size,
		 			 const int *keys, const int *vals, 
					 int *content, const int table_size,
					 int *output, int *current) {
	long  global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    long  global_threads = blockDim.x * gridDim.x;

    for (long tuple_id = global_idx; tuple_id < input_size; tuple_id += global_threads ){
		int key = keys[tuple_id];
		uint64_t location = key;
		
		// (1)
		//location = (13 ^ location ) & (table_size-1);
		//location &= (table_size-1);
		//location = location % table_size;
		// (2)
		//location = location << 1;
		//location*= HASH_FACTOR;
		location &= (table_size-1);
		
		uint64_t init_loc = location;

		for (unsigned j = 0;; j++) {
		//printf("key: %u, j:%u, loc:%u, fkey:%u\n", key, j, location, content[location*2]);
			if (key == content[location * 2]) {
				//printf("new tuple %d %d %d\n", content[location *2], content[location*2+1], tuple_id);
				// (1)
				int tmp = atomicAdd(current,3);
				output[tmp] = content[location * 2]; // key
				output[tmp+1] = content[location * 2 + 1]; // Set Value
				output[tmp+2] = vals[tuple_id]; // Set Value2
				/// (2)
				//output[tuple_id] = content[location * 2 + 1]; // Set Value
				//content[location * 2 + 1] = 1; // Set Value
				break;
			}
			
			if (content[location *2] == 0)
				break; // Empty

			if (++location == table_size)
				location = 0;

			if (location == init_loc)
				break; // Back to start
		}
	}
}


/*******************************************************************************/
/*******************************************************************************/
/*	Integer Based	*/

/*
 * Invoced by insert.
 * Return true if update worked
 */
__device__ 
bool update(int location, int key, int value, int *content) {
    int current_key = content[location * 2];
    if (current_key == 0) { 
        int old = atomicCAS(&content[location * 2], 0, key);

        if (old != 0) {
			// lost race
            return false;
        }
    } else { 
		// current position is already taken
        return false;
    }
	//printf("wrote key %d at loc %d\n", content[location * 2], location);*/
	content[location * 2 + 1] = value;
    return true;
}

// shared memory ht
// perfect hashing 
// gpu 3

__device__ 
void insert(int *content, const uint64_t table_size, const int key, const int val) {
	uint64_t location = key;

    location *= HASH_FACTOR;
	
	//int tmp = location/table_size;
    //location =  location - (table_size*tmp);
	location = location % table_size;
	//location &= (uint64_t)table_size -1;
	//location &= ~1ul;

	//int location;
	//MurmurHash_x86_32(&key, sizeof(int), 0, &location);
	//location = location % table_size;

	//printf("location: %u\n",location);
	for (int j = 0; MAX_ITERATION_ATTEMPTS > j; j++) {
		//printf("key: %u, j:%u, loc:%u\n", key, j, location);
		
		if (update(location, key, val, content))
			return;
		
		if (++location == table_size)
			location = 0;
	}
    return;
}

__global__
void build_linprobe(const int input_size, int *content, 
					 const uint64_t table_size, const int *keys, 
					 const int *vals) {
	long  global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    long  global_threads = blockDim.x * gridDim.x;

    for (long tuple_id = global_idx; tuple_id < input_size; tuple_id += global_threads ){
		insert(content, table_size, keys[tuple_id], vals[tuple_id]);
    }
}


/*
 * Invoced by insert.
 * Return true if update worked
 */
__device__ 
bool update(unsigned location, unsigned key, unsigned value, unsigned *content) {
    unsigned current_key = content[location * 2];
    if (current_key == 0) { 
        unsigned old = atomicCAS(&content[location * 2], 0, key);

        if (old != 0) {
			// lost race
            return false;
        }
    } else { 
		// current position is already taken
        return false;
    }
	//printf("wrote key %d at loc %d\n", content[location * 2], location);*/
	content[location * 2 + 1] = value;
    return true;
}

/*
 * Inserts input_size keys and values into the hash table
 * The Hashtable has to be filled with 0 if empty
 */
__global__ 
void build(const unsigned input_size, unsigned *content, 
					 const unsigned table_size, const unsigned *keys, 
					 const unsigned *vals) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) {
        unsigned location;
        unsigned key = keys[idx];
        unsigned val = vals[idx];
        MurmurHash_x86_32(&key, sizeof(unsigned), 0, &location);
        location = location % table_size;

		//printf("location: %u\n",location);
		for (unsigned j = 0; MAX_ITERATION_ATTEMPTS > j; j++) {
			//printf("key: %u, j:%u, loc:%u\n", key, j, location);
            
			if (update(location, key, val, content))
				return;
			
            if (++location == table_size)
                location = 0;
        }
    }
    return;
}


__device__ 
void insert(unsigned *content, 
					 const unsigned table_size, const unsigned key, 
					 const unsigned val) {
	unsigned location;
	MurmurHash_x86_32(&key, sizeof(unsigned), 0, &location);
	location = location % table_size;

	//printf("location: %u\n",location);
	for (unsigned j = 0; MAX_ITERATION_ATTEMPTS > j; j++) {
		//printf("key: %u, j:%u, loc:%u\n", key, j, location);
		
		if (update(location, key, val, content))
			return;
		
		if (++location == table_size)
			location = 0;
	}
    return;
}




__global__
void build_linprobe(const unsigned input_size, unsigned *content, 
					 const unsigned table_size, const unsigned *keys, 
					 const unsigned *vals) {

	const uint32_t global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    const uint32_t global_threads = blockDim.x * gridDim.x;

    for ( uint64_t tuple_id = global_idx; tuple_id < input_size; tuple_id += global_threads ){
		insert(content, table_size, keys[tuple_id], vals[tuple_id]);
    }
}

/*
 * Inserts input_size keys and values into the hash table
 * The Hashtable has to be filled with 0 if empty
 */
/*
__global__ 
void build_and_probe_sm(const unsigned table_size, 
					 const unsigned input_size_build,
					 const unsigned *keys_build, const unsigned *vals_build,
					 const unsigned input_size_probe,
					 const unsigned *keys_probe, const unsigned *vals_probe,
					 unsigned *output,
					 unsigned *current,
					 unsigned *content_) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ unsigned content[];
	
	// Build
    if (idx < input_size_probe) {
        unsigned location;
        unsigned key = keys_build[idx];
        unsigned val = vals_build[idx];
        MurmurHash_x86_32(&key, sizeof(unsigned), 0, &location);
        location = location % table_size;

	//	printf("key: %u\n",idx);
		for (unsigned j = 0; MAX_ITERATION_ATTEMPTS > j; j++) {
			 //printf("key: %u, j:%u, loc:%u\n", key, j, location);
            
			if (update(location, key, val, content))
				break;
			
            if (++location == table_size)
                location = 0;
        }
    }

	//Probe
	if (idx < input_size_probe) {
        unsigned key = keys_probe[idx];
        unsigned location;
        MurmurHash_x86_32(&key, sizeof(unsigned), 0, &location);
        location = location % table_size;
		unsigned init_loc = location;

        for (unsigned j = 0;; j++) {
		//printf("key: %u, j:%u, loc:%u, fkey:%u\n", key, j, location, content[location*2]);
			
			if (content[location * 2] == 0)
				return; // Empty

			if (key == content[location * 2]) {
				printf("%d	 %d\n",key, location );
				unsigned tmp = atomicAdd(current, 1);
				//if (tmp < table_size)
				output[tmp] = content[location * 2 + 1]; // Set Value
			}
			
            if (++location == table_size)
                location = 0;

			if (location == init_loc)
				return; // Back to start
        }
    }
    return;
}
*/

/*
 * Probes input_size keys and values with the hash table.
 * Each Inputtuple one thread
 */
__global__ 
void probe(const unsigned input_size,
		 			 const unsigned *keys, const unsigned *vals, 
					 unsigned *content, const unsigned table_size,
					 unsigned *output, unsigned *current) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_size) {
        unsigned key = keys[idx];
        unsigned location;
        MurmurHash_x86_32(&key, sizeof(unsigned), 0, &location);
        location = location % table_size;
		unsigned init_loc = location;

        for (unsigned j = 0;; j++) {
		//printf("key: %u, j:%u, loc:%u, fkey:%u\n", key, j, location, content[location*2]);
			
			if (content[location * 2] == 0)
				return; // Empty

			if (key == content[location * 2]) {
				//printf("%d	 %d\n",key, location );
				unsigned tmp = atomicAdd(current, 1);
				//if (tmp < table_size)
				output[tmp] = content[location * 2 + 1]; // Set Value
			}
			
            if (++location == table_size)
                location = 0;

			if (location == init_loc)
				return; // Back to start
        }
    }
    return;
}

/*******************************************************************************/
/*******************************************************************************/

/*
 *  Allocate Memory for 32bit key-value pairs
 */
Hashtbl::Hashtbl(HashtblConfig config) {
    CUDA_SAFE(cudaMalloc(&content, config.table_size * sizeof(unsigned) * 2))
	CUDA_SAFE(cudaMemset(content, 0, config.table_size * sizeof(unsigned) *2))
    number_of_elements = 0;
    table_size = config.table_size;
}


/*******************************************************************************/
/*******************************************************************************/
// JOIN

/*
 * Vals, keys is stored in pinned memory
 */
bool Hashtbl::Build(const unsigned input_size, unsigned *keys, unsigned *vals){
	
	// init
	unsigned *keys_d, *vals_d;
    CUDA_SAFE(cudaMalloc(&keys_d, sizeof(unsigned) * input_size))
    CUDA_SAFE(cudaMalloc(&vals_d, sizeof(unsigned) * input_size))

    CUDA_SAFE(cudaMemcpy(keys_d, keys, sizeof(unsigned) * input_size, cudaMemcpyDefault))
    CUDA_SAFE(cudaMemcpy(vals_d, vals, sizeof(unsigned) * input_size, cudaMemcpyDefault))

	// Set Blocksize
    int blockSize, minGridSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)build, 0, input_size);
    gridSize = ((long)input_size + blockSize - 1) / blockSize;

    build<<<gridSize, blockSize>>>(input_size, content, table_size, keys_d, vals_d);

	CUDA_SAFE(cudaFree(keys_d))
	CUDA_SAFE(cudaFree(vals_d))
    return true;
}

/*
 * Vals, Keys is stored in pinned memory
 */
bool Hashtbl::Probe(const unsigned input_size, unsigned *keys, unsigned *vals, 
						unsigned *output_size, unsigned *output){

	// Init
	unsigned *current_d, *output_d, *keys_d, *vals_d;
    CUDA_SAFE(cudaMalloc(&keys_d, sizeof(unsigned) * input_size))
    CUDA_SAFE(cudaMalloc(&vals_d, sizeof(unsigned) * input_size))
    CUDA_SAFE(cudaMalloc(&output_d, sizeof(unsigned) * table_size *2))
    CUDA_SAFE(cudaMalloc(&current_d, sizeof(unsigned)))

    CUDA_SAFE(cudaMemcpy(keys_d, keys, sizeof(unsigned) * input_size, cudaMemcpyDefault))
    CUDA_SAFE(cudaMemcpy(vals_d, vals, sizeof(unsigned) * input_size, cudaMemcpyDefault))

	// Set Blocksize
    int blockSize, minGridSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)probe, 0, input_size);
    gridSize = ((long)input_size + blockSize - 1) / blockSize;

	probe<<<gridSize, blockSize>>>(input_size, keys_d, vals_d, content, 
			table_size, output_d, current_d);

	/* Same result?
	unsigned *current;
    CUDA_SAFE(cudaMemcpy(current, current_d, sizeof(unsigned), cudaMemcpyDefault))
	CUDA_SAFE(cudaMemcpy(output, output_d, current * sizeof(unsigned), cudaMemcpyDefault))
	*output_size = *current;
    */
	
	CUDA_SAFE(cudaMemcpy(output, output_d, table_size * sizeof(unsigned) *2, cudaMemcpyDefault))
	*output_size = table_size *2;

	CUDA_SAFE(cudaFree(keys_d))
	CUDA_SAFE(cudaFree(vals_d))
	CUDA_SAFE(cudaFree(current_d))
	CUDA_SAFE(cudaFree(output_d))
    return true;
}

/*******************************************************************************/
/*******************************************************************************/

/*
 *  Print current state of Hashtbl.
 */
void Hashtbl::DumpHashtbl() {
    cudaThreadSynchronize();
    unsigned *content_h;
    CUDA_SAFE(cudaMallocHost(&content_h, table_size *2 * sizeof(unsigned)))
    CUDA_SAFE(cudaMemcpy(content_h, content, table_size *2* sizeof(unsigned),
                         cudaMemcpyDefault))

    printf("-- Dump Hashtbl: --\n");
    for (int i = 0; i < table_size; i++) {
        printf("key: %u - value: %u\n", content_h[i * 2], content_h[i * 2 + 1]);
    }
    printf("-------------------\n");
}

Hashtbl::~Hashtbl() { 
	CUDA_SAFE(cudaMemset(content, 0, table_size *2*sizeof(unsigned)))
	//cudaFree(content); 
	cudaFree(content); 

}
