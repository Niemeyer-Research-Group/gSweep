# Simple makefile for all test problems for 1D Swept Rule

# LDFLAGS := -lglut -lGL -lGLU -lGLEW

# It will put the compiled programs in the bin subdirectory within the directory
# containing the makefile

PROJECT_DIR=$(shell pwd)
EQPATH:=$(abspath equations)

INC_PARAMS=$(INC:%=-I%)

EQA:=$(wildcard $(EQPATH)/*.h)

EQB:=$(filter-out $(EQPATH)/eqHead.h, $(EQA))

EQC:=$(foreach eq, $(EQB), $(basename $(notdir $(eq))))

OBJS:=$(foreach eq, $(EQC), $(OUT_DIR)/$(eq).o)

all: $(OBJS) 

$(OBJS): kernel.cu equation.cu 
	nvcc $^ -o $@ -include $(EQB)  $(CUDAFLAGS) $(INC_PARAMS) 



