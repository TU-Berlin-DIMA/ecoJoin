// FILE: benchmark_helper_lib.h

#define CUDA_SAFE(func)    {                                              \
        cudaError err = func;                                             \
        if (cudaSuccess != err) {                                         \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }


#define  CREATE_STREAMS   \
        cudaStream_t *streams =                 \
        (cudaStream_t *)malloc(streamNumber * sizeof(cudaStream_t)); \
        for (int i = 0; i < streamNumber; i++) \
        CUDA_SAFE(cudaStreamCreate(&streams[i]))

#define DESTROY_STREAMS \
        for (int i = 0; i < streamNumber; i++) \
        CUDA_SAFE(cudaStreamDestroy(streams[i]))

#define CREATE_KERNEL_SIZE(func, num) \
    int blockSize, minGridSize, gridSize;\
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,\
                                       func, 0,\
                                       num);\
    gridSize = (num + blockSize - 1) / blockSize;

#define INIT_MEASURE \
    cudaEvent_t start, stop; \
    unsigned avgIteration = 0; \
    double avgIterationSum = 0;

#define START_MEASURE(N) \
    avgIteration = 0; \
    avgIterationSum = 0; \
    for (; avgIteration < N; avgIteration++) { \
    	initializeEvents(&start, &stop); 

#define END_MEASURE(N) \
   	avgIterationSum += finalizeEvents(start, stop);\
    } \
    printf("%f ms\n", avgIterationSum / N);

#define INIT_MEASUREX \
   Timestamp begin, end;

#define START_MEASUREX \
   begin = getTimestamp();

#define END_MEASUREX \
   end = getTimestamp(); \
   std::cout << "--x: " << double(end - begin) / (1000 * 1000) \
              << " ms" << std::endl;


#define INIT_ASYNC(blockByte_, datatype) \
    unsigned elementsPerBlock = blockByte_ / sizeof(datatype); \
    unsigned blockNumber = sizeByte / blockByte_; \
    unsigned blockByte = blockByte_; \
    int currentStream = 0;

#define EXE_ASYNC(func, args, input_p, output_p, input_d, output_d, shared) \
    for (int i = 0; i < blockNumber; i++) { \
         int offset = i * elementsPerBlock; \
         CUDA_SAFE(cudaMemcpyAsync(&input_d[offset], &input_p[offset], blockByte, \
                                   cudaMemcpyHostToDevice, \
                                   streams[currentStream])) \
         func<<<gridSize, blockSize,shared, \
                                 streams[currentStream]>>>args; \
         CUDA_SAFE(cudaMemcpyAsync(&output_p[offset], &output_d[offset], \
                                   blockByte, cudaMemcpyDeviceToHost, \
                                   streams[currentStream])); \
         currentStream = ++currentStream % streamNumber; \
     }



void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop);

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop);

template <typename T>
__global__ void increment(T *array_d, unsigned N);
__global__ void increment(float *array_d, unsigned N);

bool checkIncrementResult(unsigned *input, unsigned *output, unsigned sizeInt);
bool checkIncrementResult(unsigned *input, unsigned *output, unsigned sizeInt, unsigned iterations);
