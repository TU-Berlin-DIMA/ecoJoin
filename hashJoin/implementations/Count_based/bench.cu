/*
 *	Count-based Streaming Join
 *
 *
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <thread>
#include <vector>

#include "benchmark_helper_lib_v5.h"
#include "hashtbl.h"

struct tuple {
  uint32_t key;
  uint64_t timestamp;
  uint32_t stream;
  uint32_t value;
};

struct result_tuple {
  uint32_t key;
  uint64_t timestamp;
  uint32_t stream;
  uint32_t value_1;
  uint32_t value_2;
};

int window_size = 0;
int *current_d;
result_tuple *output;
// tuple *build_tuple, probe_tuple;
tuple *content_stream1, *content_stream2;
std::vector<uint64_t> latencies;
std::chrono::time_point<std::chrono::steady_clock> start_name, end_name;

#define MAX_ITERATION_ATTEMPTS 1000000
__global__ void build_linprobe_lsb(const int input_size, tuple *content, const uint64_t buckets,
                                   const tuple *tupl) {
  long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  long global_threads = blockDim.x * gridDim.x;
  for (long tuple_id = global_idx; tuple_id < input_size; tuple_id += global_threads) {
    // printf("%d\n", tuple_id);
    uint64_t currentBucket = tupl[tuple_id].key;
    // (1)
    // currentBucket = (13 ^ currentBucket) & (buckets- 1);
    // currentBucket = currentBucket % buckets;
    // (2)
    // currentBucket = currentBucket << 1;
    // currentBucket *= HASH_FACTOR;
    currentBucket &= buckets - 1;
    for (int j = 0; MAX_ITERATION_ATTEMPTS > j; j++) {
      // printf("key: %u, j:%u, loc:%u\n",keys[tuple_id], j,
      // currentBucket );
      int current_key = content[currentBucket].key;
      if (current_key == 0) {
        // content[location * 2] = keys[tuple_id];
        int old = atomicCAS(&content[currentBucket].key, 0, tupl[tuple_id].key);
        if (old == 0) {
          content[currentBucket].value = tupl[tuple_id].value;
          // printf("key: %d, value: %d\n", content[currentBucket].key,
          // content[currentBucket].value);
          break;
        }
      }
      if ((++currentBucket) * 2 == buckets)
        currentBucket = 0;
    }
  }
}

__global__ void probe_linprobe_lsb(const int input_size, tuple *tupl, tuple *content,
                                   const int table_size, result_tuple *output, int *current) {
  long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  long global_threads = blockDim.x * gridDim.x;

  for (long tuple_id = global_idx; tuple_id < input_size; tuple_id += global_threads) {
    int key = tupl[tuple_id].key;
    uint64_t location = key;

    location &= (table_size - 1);

    uint64_t init_loc = location;
    for (unsigned j = 0;; j++) {
      // printf("key: %d, content: %d\n", key, content[location].key);
      if (key == content[location].key) {
        int tmp = atomicAdd(current, 1);
        output[tmp].key = content[location].key;       // key
        output[tmp].value_1 = content[location].value; // Set Value
        output[tmp].value_2 = tupl[tuple_id].value;    // Set Value2
        output[tmp].timestamp = tupl[tuple_id].timestamp;    // Set Timestamp
        break;
      }

      if (content[location].value == 0)
        break; // Empty

      if (++location == table_size)
        location = 0;

      if (location == init_loc)
        break; // Back to start
    }
  }
}

void startStream(tuple *records0, int rows) {
  tuple *new_tuple;
  int epoch = 0;
  int counter = 0;
  int input_size = 1;
  cudaStream_t stream;
  CUDA_SAFE(cudaStreamCreate(&stream));
  START_M()
  for (int i = 0; i < rows; i++) {
    new_tuple = &records0[i];

    // Fill with current time
    auto now = std::chrono::steady_clock::now();
    auto nanos =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    new_tuple->timestamp = (uint64_t)nanos;

    /* ------------------------------------------------------------------------*/
    /* ------------------------------------------------------------------------*/

    counter += input_size;
    if (counter >= window_size) {
      // std::cout << "reset window\n";
      CUDA_SAFE(cudaMemset(content_stream1, 0, window_size * sizeof(tuple)));
      CUDA_SAFE(cudaMemset(content_stream2, 0, window_size * sizeof(tuple)));
      counter = 0;
    }

    tuple *content;
    if (new_tuple->stream == 0) {
      build_linprobe_lsb<<<input_size, 1, 0, stream>>>(1, content_stream1, window_size, new_tuple);
      CUDA_SAFE(cudaStreamSynchronize(stream));

      probe_linprobe_lsb<<<input_size, 1, 0, stream>>>(1, new_tuple, content_stream2, window_size,
                                                       output, current_d);
      CUDA_SAFE(cudaStreamSynchronize(stream));

    } else {
      build_linprobe_lsb<<<input_size, 1, 0, stream>>>(1, content_stream2, window_size, new_tuple);
      CUDA_SAFE(cudaStreamSynchronize(stream));

      probe_linprobe_lsb<<<input_size, 1024, 0, stream>>>(1, new_tuple, content_stream1,
                                                          window_size, output, current_d);
      CUDA_SAFE(cudaStreamSynchronize(stream));
    }

    //std::cout << current_d[0] << "\n";
    if (current_d[0] == 1) {
      auto now = std::chrono::steady_clock::now();
      auto nanos =
          std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

      latencies.push_back((uint64_t)nanos - output[0].timestamp);
    }

    CUDA_SAFE(cudaMemset(current_d, 0, sizeof(int)));
    if ((i % 10000) == 0)
      printf("%d\n", i + rows * epoch);
    if (++i == rows - 1) {
      i = 1; // 0 is first tuple
      epoch++;
    }
  }
}

void CtrlChandler(int _) {
  printf("\n");

  int i = 1;
  uint64_t std = 0;
  for (auto a : latencies) {
    std += a;
    i++;
  }
  std = std / i;
  std::cout << "latency: " << std << " ns\n";

  end_name = std::chrono::steady_clock::now();
  int s = std::chrono::duration_cast<std::chrono::seconds>(end_name - start_name).count();
  std::cout << "elapsed time: " << s << "s\n";

  std::cout << "(result tuples) #: " << latencies.size() << " \n";
  std::cout << "(result tuples) #: " << latencies.size() / s << " #/s\n";
  float kb = (float)(latencies.size() * 20 /*B*/) / (1000);
  std::cout << "(result tuples) kb: " << kb << " kb\n";
  std::cout << "(result tuples) kb/s: " << (float)kb / s << " kb/s\n";
  exit(0);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: bench [window_size]");
    exit(0);
  }
  window_size = atoi(argv[1]);
  signal(SIGINT, CtrlChandler);

  // Input Buffer
  const int buffersize = 100 * 100; //* 100;
  tuple *records0;
  cudaHostAlloc((void **)&records0, buffersize * sizeof(tuple), cudaHostAllocDefault);
  for (int i = 0; i < buffersize; i++) {
    records0[i].key = rand() % 10000 + 1;
    records0[i].value = rand() % 10000 + 1;
    records0[i].timestamp = 0;
    records0[i].stream = rand() % 2; // Stream1 or Stream2
  }
  std::cout << "Input buffer created\n";

  // cudaHostAlloc((void **)&probe_tuple, rows * sizeof(tuple), cudaHostAllocDefault);
  // cudaHostAlloc((void **)&build_tuple, rows * sizeof(tuple), cudaHostAllocDefault);

  CUDA_SAFE(cudaMalloc((void **)&content_stream1, window_size * sizeof(tuple)));
  CUDA_SAFE(cudaMalloc((void **)&content_stream2, window_size * sizeof(tuple)));

  CUDA_SAFE(cudaHostAlloc((void **)&current_d, sizeof(int), cudaHostAllocDefault));
  CUDA_SAFE(
      cudaHostAlloc((void **)&output, buffersize * sizeof(result_tuple), cudaHostAllocDefault));

  startStream(records0, buffersize);
}
