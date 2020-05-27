HASH     = 0

TARGET   = gpu_stream

CUFLAGS  = -std=c++11 -gencode arch=compute_53,code=sm_53 -g -Xcompiler=-fopenmp
CFLAGS   = -std=c++11 -I. -g -I/usr/local/cuda/include -L/usr/local/cuda/lib64

LFLAGS   =  -gencode arch=compute_53,code=sm_53 -g -lcpufreq -lgomp -ltbb

NVCC     = /usr/local/cuda/bin/nvcc
CC       = g++
LINKER   = /usr/local/cuda/bin/nvcc

SRCDIR   = src
OBJDIR   = obj
BINDIR   = bin

CPPSOURCES   := $(wildcard $(SRCDIR)/*.cpp)
CPPINCLUDES  := $(wildcard $(SRCDIR)/*.hpp)

CSOURCES   := $(wildcard $(SRCDIR)/*.c)
CINCLUDES  := $(wildcard $(SRCDIR)/*.h)

CUSOURCES  := $(wildcard $(SRCDIR)/*.cu)
CUINCLUDES := $(wildcard $(SRCDIR)/*.hu)

COBJECTS   := $(CSOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
CPPOBJECTS := $(CPPSOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
CUOBJECTS  := $(CUSOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
rm         = rm -f

$(BINDIR)/$(TARGET): $(COBJECTS) $(CUOBJECTS) $(CPPOBJECTS) 
	@echo $(LINKER) $(COBJECTS) $(CUOBJECTS) $(CPPOBJECTS) $(LFLAGS) -o $@
	@$(LINKER) $(COBJECTS) $(CUOBJECTS) $(CPPOBJECTS) $(LFLAGS) -o $@
	@echo "Linking complete!"

$(COBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.c
	@echo $(CC) $(CFLAGS) -c $< -o $@
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiled "$<" successfully!"

$(CPPOBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@echo $(CC) $(CFLAGS) -c $< -o $@
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiled "$<" successfully!"

$(CUOBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cu
	@echo $(NVCC) $(CUFLAGS) -c $< -o $@
	@$(NVCC) $(CUFLAGS) -c $< -o $@
	@echo "Compiled "$<" successfully!"

.PHONY: clean
clean:
clear:
	rm obj/*
	rm bin/*
