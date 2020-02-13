#!/bin/bash
{ ../bin/gpu_stream $@;} &
wait -n
kill -- -$$
