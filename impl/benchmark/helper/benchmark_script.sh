#!/bin/bash
{ ../bin/gpu_stream $@;} &
{ mpstat  -P ALL 1  > cpu_usage.txt;} &
wait -n
kill -- -$$
