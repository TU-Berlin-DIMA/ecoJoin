#!/bin/bash
{ ../bin/gpu_stream -n 100000 -N 100000 -r 4000 -R 4000 -s 2 -S 1 -p gpu;} &
{ ./gpu_monitor.sh; } &
wait -n
kill -- -$$
