# Simple makefile for all test problems for 1D Swept Rule

CUDAFLAGS = -gencode arch=compute_35,code=sm_35 -lm -restrict -O3 --ptxas-options=-v -I./utilities --std=c++11
CUDAFLAGSDOUBLE := $(CUDAFLAGS) -DREAL=double -DREALtwo=double2 -DREALthree=double3 -DREALfour=double4
# LDFLAGS := -lglut -lGL -lGLU -lGLEW

# It will put the compiled programs in the bin subdirectory within the directory
# containing the makefile
PROJECT_DIR = $(shell pwd)
OUT_DIR := ./bin

all: $(OUT_DIR)/waveReg $(OUT_DIR)/waveCoop

$(OUT_DIR)/waveReg: regkernel1D.cu mainEntry.cpp utilities/jsoncpp.cpp mainGlobals.h equation.h 
	nvcc -c $< -o $@ $(CUDAFLAGS)

$(OUT_DIR)/waveCoop: coopkernel1D.cu mainEntry.cpp utilities/jsoncpp.cpp mainGlobals.h equation.h 
	nvcc -c $< -o $@ $(CUDAFLAGS)

# -------------------------

# $(OUT_DIR)/waveReg: $(OUT_DIR)/wvMain.o $(OUT_DIR)/waveR.o
# 	nvcc $^ -o $@ $(LDFLAGS)

# $(OUT_DIR)/waveCoop: $(OUT_DIR)/wvMain.o $(OUT_DIR)/waveCo.o
# 	nvcc $^ -o $@ $(LDFLAGS)


