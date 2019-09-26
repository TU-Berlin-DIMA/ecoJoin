#include <bitset>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <signal.h>
#include <stdlib.h>
#include <thread>
#include <vector>

#define DEBUG 0

#define DEBUG_P(pr)                                                                                \
  if (DEBUG)                                                                                       \
    std::cout << pr << "\n";

#define CUDA_SAFE(call)                                                                            \
  do {                                                                                             \
    cudaError_t err = call;                                                                        \
    if (cudaSuccess != err) {                                                                      \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__, __LINE__,              \
              cudaGetErrorString(err));                                                            \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define START_M(name)                                                                              \
  start_name = std::chrono::system_clock::now();

#define END_M(name)                                                                                \
  end_name = std::chrono::system_clock::now();                                                     \
  int elapsed_seconds_name =                                                                       \
      std::chrono::duration_cast<std::chrono::milliseconds>(end_name - start_name).count();        \
  std::time_t end_time_name = std::chrono::system_clock::to_time_t(end_name);                      \
  std::cout << "elapsed time: " << elapsed_seconds_name << "ms\n";


std::chrono::time_point<std::chrono::system_clock> start_name, end_name;

enum Stream { stream0, stream1 };

struct tuple {
  uint32_t key;
  uint64_t timestamp;
  uint32_t stream;
  uint32_t value;
};

struct resultTuple {
  uint32_t key;
  uint64_t timestamp;
  uint32_t left_value;
  uint32_t right_value;
};

tuple *compareTuples_s1_inter, *compareTuples_s1_comp;
__device__ size_t currentFIFO_s1 = 0;
size_t currentFIFO_s1_inter = 0;

tuple *compareTuples_s2_inter, *compareTuples_s2_comp;
__device__ size_t currentFIFO_s2 = 0;
size_t currentFIFO_s2_inter = 0;

std::vector<tuple> compareTuples_h;
int *compare_output_s1, *compare_output_s2;
int *compare_output_prev;
int etpw, gridsize, blocksize;
std::ofstream myfile;

__host__ __device__ void printRecord(tuple rec) {
#ifdef DEBUG
  printf("key: %lu timestamp: %llu value: %lu\n", (unsigned long)rec.key,
         (unsigned long long)rec.timestamp, (unsigned long)rec.value);
#endif
}

const int timeLimit = 100000; // ns

std::vector<uint64_t> latencies;

/*
 * Compare Kernel
 * Encodes the result as 32 bit Integer
 */
__global__ void compare_kernel_ipt(tuple input, int *output, size_t etpw, tuple *compareTuples) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  int z = 0;
  if ((idx + 1) * 32 < etpw) {
#pragma unroll
    for (int i = 0; i < 32; i++) {
      if (((input.timestamp - compareTuples[idx + (32 * i)].timestamp < timeLimit) ||
           (compareTuples[idx + (32 * i)].timestamp - input.timestamp < timeLimit)) &&
          (input.key == compareTuples[idx + (32 * i)].key)) {
        printRecord(input);
        z = z | 1 << i;
      }
    }
  } else if (idx * 32 < etpw) {
    for (int i = 0; i < etpw - idx * 32; i++) {
      if (((input.timestamp - compareTuples[idx + (32 * i)].timestamp < timeLimit) ||
           (compareTuples[idx + (32 * i)].timestamp - input.timestamp < timeLimit)) &&
          (input.key == compareTuples[idx + (32 * i)].key)) {
        printRecord(input);
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
__global__ void add_new_tuple_device(tuple new_tuple, Stream stream, size_t etpw,
                                     tuple *compareTuples) {
  if (stream == stream1) {
    compareTuples[currentFIFO_s1].key = new_tuple.key;
    compareTuples[currentFIFO_s1].value = new_tuple.value;
    compareTuples[currentFIFO_s1].timestamp = new_tuple.timestamp;
    printRecord(new_tuple);
    if (++currentFIFO_s1 == etpw)
      currentFIFO_s1 = 0;
  } else {
    compareTuples[currentFIFO_s2].key = new_tuple.key;
    compareTuples[currentFIFO_s2].value = new_tuple.value;
    compareTuples[currentFIFO_s2].timestamp = new_tuple.timestamp;
    printRecord(new_tuple);
    if (++currentFIFO_s2 == etpw)
      currentFIFO_s2 = 0;
  }
}

__global__ void print_state(const tuple *compareTuples, int etpw) {
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
std::vector<resultTuple> interprete(tuple input, int *bitmap, Stream stream) {
  DEBUG_P("Current rec ")
  printRecord(input);

  // Add locally
  if (stream == stream1) {
    compareTuples_s1_inter[currentFIFO_s1_inter].key = input.key;
    compareTuples_s1_inter[currentFIFO_s1_inter].value = input.value;
    compareTuples_s1_inter[currentFIFO_s1_inter].timestamp = input.timestamp;
    currentFIFO_s1_inter++;
    if (currentFIFO_s1_inter == etpw)
      currentFIFO_s1_inter = 0;
  } else {
    compareTuples_s2_inter[currentFIFO_s2_inter].key = input.key;
    compareTuples_s2_inter[currentFIFO_s2_inter].value = input.value;
    compareTuples_s2_inter[currentFIFO_s2_inter].timestamp = input.timestamp;
    currentFIFO_s2_inter++;
    if (currentFIFO_s2_inter == etpw)
      currentFIFO_s2_inter = 0;
  }

  std::vector<resultTuple> result;
  for (int i = 0; i < etpw; i = i + 32) {
    if (bitmap[i / 32] == 0) { // first check
      continue;
    } else {
#pragma unroll
      for (int k = 0; k < 32; k++) {
        int j = i + k;
        if (std::bitset<32>(bitmap[j / 32]).test(j % 32)) { // fine-grained check
          // bitmap[i / 32] = bitmap[j / 32] & ~(1 << (j % 32));  //
          // ith bit = 0
          int z = (j / 32) + ((j % 32) * 32);
          resultTuple r;
          if (stream == stream1) {
            DEBUG_P("Match  ")
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

void print_result(std::vector<resultTuple> result) {
  for (auto a : result) {
    std::cout << "match  newtuple (" << a.timestamp << ", " << a.key << ", " << a.right_value
              << ", " << a.left_value << ") \n";

	// Current timestamp 
	auto now = std::chrono::high_resolution_clock::now();
	auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

	latencies.push_back((uint64_t) nanos - a.timestamp);
  }
}

void write_result(std::vector<resultTuple> result) {
  for (auto a : result)
    myfile << "match  newtuple (" << a.timestamp << ", " << a.key << ", " << a.right_value << ", "
           << a.left_value << ") \n";
}

void startStream(const tuple *records0, int rows) {
  START_M(_)
  Stream stream_prev;
  tuple tuple_prev;
  tuple new_tuple;

  int epoch = 0;
  for (int i = 0; i < rows; i++) {
    new_tuple = records0[i];

    // Fill with current time
    auto now = std::chrono::high_resolution_clock::now();
    auto nanos =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    new_tuple.timestamp = (uint64_t)nanos;

    if (new_tuple.stream == 0) {
      compare_kernel_ipt<<<blocksize, gridsize>>>(new_tuple, compare_output_s1, etpw,
                                                  compareTuples_s2_comp);
      add_new_tuple_device<<<1, 1>>>(new_tuple, stream0, etpw, compareTuples_s1_comp);

      // Start interpretation of prev tuple while execution
      if (i != 0)
        print_result(interprete(tuple_prev, compare_output_prev, stream_prev));

      cudaStreamSynchronize(0);

      // Save prev setup.
      std::memcpy(compare_output_prev, compare_output_s1, sizeof(int) * ((etpw / 32) + 1));
      stream_prev = stream0;
      tuple_prev = new_tuple;
    }
    if (new_tuple.stream == 1) {
      compare_kernel_ipt<<<blocksize, gridsize>>>(new_tuple, compare_output_s2, etpw,
                                                  compareTuples_s1_comp);
      add_new_tuple_device<<<1, 1>>>(new_tuple, stream1, etpw, compareTuples_s2_comp);

      // Start interpretation of prev tuple while execution
      if (i != 0)
        print_result(interprete(tuple_prev, compare_output_prev, stream_prev));

      cudaStreamSynchronize(0);

      // Save prev setup.
      std::memcpy(compare_output_prev, compare_output_s2, sizeof(int) * ((etpw / 32) + 1));
      stream_prev = stream1;
      tuple_prev = new_tuple;
    }
    if ((i % 10000) == 0)
      printf("%d\n", i + rows * epoch);
    if (++i == rows - 1) {
      i = 1; // 0 is first tuple
      epoch++;
    }
  }
  print_result(interprete(tuple_prev, compare_output_prev, stream_prev));
  END_M(_)
}

void CtrlChandler(int _){
  printf("\n");
	
  int i = 0;
  uint64_t std = 0;
  for (auto a : latencies) {
	  std += a;
	  i++;
  }
  std = std / i;
  std::cout << "latency: " << std << " ns\n";
  
  end_name = std::chrono::system_clock::now();                                                     
  int s = std::chrono::duration_cast<std::chrono::seconds>(end_name - start_name).count();
  std::cout << "elapsed time: " << s << "s\n";
  
  float kb  = (float) (latencies.size() * 20 /*B*/) / (1000);
  std::cout << "(result tuples) kb: " << kb << " kb\n";
  std::cout << "(result tuples) kb/s: " << (float) kb/s << " kb/s\n";
  exit(0);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: hellsjoin [window_size]\n");
    exit(0);
  }
  etpw = atoi(argv[1]);
  blocksize = 32;               // Number of threads per block
  gridsize = (etpw / 1024) + 1; // Number of blocks

  signal(SIGINT, CtrlChandler);

  std::cout << "Blocksize: " << blocksize << " Gridsize: " << gridsize << "\n";

  myfile.open("result.csv");

  // Compare Tuple Buffer - Device
  CUDA_SAFE(cudaHostAlloc((void **)&compareTuples_s1_comp, sizeof(tuple) * etpw, 0));
  CUDA_SAFE(cudaHostAlloc((void **)&compareTuples_s2_comp, sizeof(tuple) * etpw, 0));

  // Compare Tuple Buffer - Host
  compareTuples_s1_inter = (tuple *)calloc(etpw, sizeof(tuple));
  compareTuples_s2_inter = (tuple *)calloc(etpw, sizeof(tuple));

  // Compare Kernel Output
  CUDA_SAFE(cudaHostAlloc((void **)&compare_output_s1, sizeof(int) * ((etpw / 32) + 1), 0));
  CUDA_SAFE(cudaHostAlloc((void **)&compare_output_s2, sizeof(int) * ((etpw / 32) + 1), 0));
  CUDA_SAFE(cudaHostAlloc((void **)&compare_output_prev, sizeof(int) * ((etpw / 32) + 1), 0));

  // Input Buffer
  const int buffersize = 100 * 100 * 100;
  tuple *records0 = (tuple *)malloc(sizeof(tuple) * buffersize);
  for (int i = 0; i < buffersize; i++) {
    records0[i].key = rand() % 100 + 1;
    records0[i].value = rand() % 100 + 1;
    records0[i].timestamp = 0;
    records0[i].stream = rand() % 2; // Stream1 or Stream2
  }
  std::cout << "Input buffer created\n";

  // startManualTest();
  startStream(records0, buffersize);

  myfile.close();
}
