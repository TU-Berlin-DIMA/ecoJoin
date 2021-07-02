# An Energy-Efficient Stream Join for the Internet of Things
This repository contains the code for ecoJoin, an energy-efficient stream join for the IoT. EcoJoin 
reduces the power consumtion of stream joins by optimizing three orthogonal dimensions:
workload characteristics, computational efficiency, and heterogeneous hardware. For further information 
we refer to our [paper](https://www.nebula.stream/paper/adrian_ecoJoin_damon2021.pdf) published 
at [DaMoN 2021](https://sites.google.com/view/damon2021). 

EcoJoin is part of the joint research project NebulaStream between the DIMA group at TU Berlin and the DFKI IAM group.
NebulaStream is the first general purpose, end-to-end data management system for the IoT.
To learn more about NebulaStream, please visit [https://www.nebula.stream](https://www.nebula.stream).

## Usage

### Dependencies
- CUDA (tested with CUDA 10)
- OpenMP
- Cpufreq (for frequency scaling)
- C++11 support

### Build the Source Code
CPU implementation:

	cd ./impl && make -j

GPU implementation:

	cd ./impl_gpu && make -j

### Run a Query
EcoJoin currently supports synthetic generated data.

Example Query:

	./impl/bin/ecoJoin -p ht_cpu4 -n 15728640 -N 15728640 -R 131072 -r 131072 -w 10 -W 10 -b 8192 -B 8192 -c 36000 -O bench.csv

This will start the 4 core CPU version of ecoJoin on `15728640` tuples with an input rate of `131072` tuples per second for both streams (resulting total runtime of 2 min).
It will run with a window size of `10` seconds, process tuples in `8192` tuple batches and uses a cleanup threshold of `36000` tuples.
 

For a list of all commands you can use `--help`. Output for `./impl_gpu/bin/ecoJoin --help`:
```
Usage:
  -n NUM   number of tuples to generate for stream R
  -N NUM   number of tuples to generate for stream S
  -O FILE  file name to write result information to
  -r RATE  tuple rate for stream R (in tuples/sec)
  -R RATE  tuple rate for stream S (in tuples/sec)
  -w SIZE  window size for stream R (in seconds)
  -W SIZE  window size for stream S (in seconds)
  -p []  processing mode ()
  -T enable sleep time window
  -t sleep control in worker
  -b NUM  batchsize for stream R
  -B NUM  batchsize for stream S
  -g NUM  GPU gridsize
  -G NUM  GPU blocksize
  -f enable frequency by stream join
  -e end when worker ends
```

For energy measures using a ARM Energy Probe take a look at: [conducting an energy measure](Energy-Measure.md)

## Publication
**Abstract:**

The Internet of Things (IoT) combines large data centers with (mobile, networked) edge devices that are constrained both in compute power and energy budget. Modern edge devices contribute to query processing by leveraging accelerated processing units with multi-core CPUs or GPUs. Therefore, data processing in the IoT presents the challenges of 1) minimizing the energy consumed while sustaining a given query throughput, and 2) processing increasingly complex queries within a given energy budget.
In this paper, we investigate how modern edge devices can reduce the energy requirements of stream joins as a common data processing operation. We explore three dimensions to save energy: workload characteristics, computational efficiency, and heterogeneous hardware. Based on our findings, we propose the ecoJoin that 1) reduces energy consumption by 81% at a given join throughput, and 2) enables scaling the throughput by two orders-of-magnitude within a given energy budget.

- Paper: [An Energy-Efficient Stream Join for the Internet of Things](https://www.nebula.stream/paper/adrian_ecoJoin_damon2021.pdf)
