# Simple makefile for all test problems for gSweep

CODE_DIR:=oneD # twoD

export OUT_DIR:=$(abspath bin)

export NVCC:=$(shell which nvcc) 
INN:=utilities $(CODE_DIR)
export INC:=$(foreach ipath, $(INN), $(abspath $(ipath)))
export CUDAFLAGS = -gencode arch=compute_60,code=sm_60 -lm -restrict -O3 --ptxas-options=-v --std=c++11
.PHONY: swepts echoes #$(EXECUTABLES)

# $(EXECUTABLES): %: $(OUT_DIR)/%.so 
# 	$(NVCC) $^ -o $(OUT_DIR)/$@ 


swepts:
	$(MAKE) -C $(CODE_DIR)

echoes:
	@echo $(OUT_DIR) $(INC)
# main.o: gsmain.cpp $(CODE_DIR)/kernel.h $(CODE_DIR)/equation.h
# 	$(NVCC) $< -o $(OUT_DIR)/$@ $(CUDAFLAGS) -I$(INC) 

# build:
#     @mkdir -p bin

# clean:
