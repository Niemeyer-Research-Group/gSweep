# Simple makefile for all test problems for 1D Swept Rule

CUDAFLAGS = -gencode arch=compute_35,code=sm_35 -lm -restrict -O3 -Xcompiler -fopenmp --ptxas-options=-v
CUDAFLAGSDOUBLE := $(CUDAFLAGS) -DREAL=double -DREALtwo=double2 -DREALthree=double3 -DREALfour=double4
LDFLAGS := -lglut -lGL -lGLU -lGLEW

# It will put the compiled programs in the bin subdirectory within the directory
# containing the makefile
PROJECT_DIR = $(shell pwd)
OUT_DIR := ./bin

all: $(OUT_DIR)/waveCoop $(OUT_DIR)/waveReg 

$(OUT_DIR)/waveReg: $(OUT_DIR)/wvMain.o $(OUT_DIR)/waveR.o
	nvcc $^ -o $@ $(LDFLAGS)

$(OUT_DIR)/waveCoop: $(OUT_DIR)/wvMain.o $(OUT_DIR)/waveCo.o
	nvcc $^ -o $@ $(LDFLAGS)

# Grid groups require relocatable device code
$(OUT_DIR)/waveR.o: regKernel.cu regKernel.h waveConsts.h
	nvcc $(CUDAFLAGS) -c $< -o $@ 

$(OUT_DIR)/waveCo.o: coopKernel.cu coopKernel.h waveConsts.h
	nvcc $(CUDAFLAGS) -dc -c $< -o $@ 

$(OUT_DIR)/wvMain.o: mainEntry.cpp interop.h mainGlobals.h
	nvcc $(CUDAFLAGS) -c $< -o $@ 

