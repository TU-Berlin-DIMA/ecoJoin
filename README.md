# hashJoin

## implementations
Benchmarks with hash join variations. 32bit key, 32bit values.

Hashing Methods:
probe linprobe lsb: Least Significant Bit Hashing.

### sequential
Std implementation with sequential keys, non-shuffled tuples
### random
Build and probe tuples are shuffled
### pipeline
Build kernel is overlapped with DtoH and HtoD transfer.
### sharedMem
Build and probe in shared memory (not working)
### bindings
Files linked by the other implementations




# hellsJoin

## implementations
HellsJoin implementations.
### HellsJoin
reimplementation of hellsjoin

### HellsJoin batch compiles
hellsjoin with batchwise processing


### Experimental implementations:
### HellsJoin less streams
hellsjoin impl with threadpool

### HellsJoin new streams
hellsjoin impl with different execution order

### HellsJoin pre app
hellsjoin impl with different execution order

### HellsJoin cpu
hellsjoin impl on cpu. based on Phillips work.

## nestedloopbench
Runtime measurement of Hells join compare kernel

## computationBenchmark
Measurements of HellsJoin compare kernel variations.
Different allocation types, batch execution, shared memory usage.



# transfer benchmark
Benchmarks for measuring the transfer runtime.
Two memory regions where _in_ is copied into _out_ using a copy inside a gpu kernel.
One benchmark measurement includes: the copy of _in_ from host to device (HtoD), the copy from _in_ to _out_ on the device, and the copy of _out_ from device to host (DtoH).
## host
Both memory regions are in pinned host memory.
## device
Both memory regions are in device memory.
## cas
Copy using compare and swap operation. Both memory regions are in device memory.
## cpu
Referernce Measure using the cpu: Copy with the cpu on pinned host memory.
## deviceAndCopy
Both memory regions are in pinned host memory. Copy between host and device is performed using cudaMemcpy call.
## pipeline
HtoD, kernel and DtoH are overlapped. Both memory regions are in pinned host memory.
## block
Transfer blockwise
## shared memory benchmark
Benchmark for measuring the shared memory vs. global memory runtime.
