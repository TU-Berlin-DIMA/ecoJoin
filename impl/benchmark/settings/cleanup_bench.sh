#!/bin/sh
for VARIABLE in 0 25000 50000 75000 100000 125000 150000 175000 200000 225000 250000
do
	echo $VARIABLE
	../../bin/gpu_stream -n 62914560 -N 62914560 -w 10 -W 10 -R 524288 -r 524288 -p ht_cpu4 -b 1024 -B 1024 -O bench.csv -c $VARIABLE
done
