#
# Copyright 2011-2015 NVIDIA Corporation. All rights reserved
# 
INCLUDES=-I/usr/local/cuda/extras/CUPTI/include

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):../../libWin32:../../libx64
    LIBS= -lcuda -L ../../libWin32 -L ../../libx64 -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):../../lib
        LIBS= -Xlinker -framework -Xlinker cuda -L ../../lib -lcupti
    else
        export LIBRARY_PATH := $(LIBRARY_PATH):/usr/local/cuda/extras/CUPTI/lib64
        LIBS= -lcuda -L/usr/local/cuda/extras/CUPTI/lib64/ -lcupti
    endif
    OBJ = o
endif

nvml_example: nvml_example.$(OBJ)
	/usr/local/cuda/bin/nvcc -o $@ $^ $(LIBS) -I/usr/local/cuda/samples/common/inc/ -lnvidia-ml -L/usr/lib -I/usr/local/cuda-9.2/targets/x86_64-linux/include -L/usr/local/cuda-9.2/targets/x86_64-linux/lib -lcublas


nvml_example.$(OBJ): nvml_example.cu nvmlClass.h
	/usr/local/cuda/bin/nvcc -c $(INCLUDES) $< -I/usr/local/cuda/samples/common/inc/ -lnvidia-ml -L/usr/lib

run: nvml_example
	./$<

clean:
	rm -f src/nvml_example.cu src/nvml_example.$(OBJ)
