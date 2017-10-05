/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

//COMPILE LINE!
// nvcc -o ./bin/KSOut KS1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -restrict -Xcompiler -fopenmp --ptxas-options=-v

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <iostream>
#include <ostream>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>

#define GPUNUM          0

#ifndef REAL
	#define REAL        float
	#define ONE         1.f
	#define TWO         2.f
	#define FOUR        4.f
	#define SIX			6.f
#else
	#define ONE         1.0
	#define TWO         2.0
	#define FOUR        4.0
	#define SIX			6.0
#endif

using namespace std;

const REAL dx = 0.5;

struct discConstants{

	REAL dxTimes4;
	REAL dx2;
	REAL dx4;
	REAL dt;
	REAL dt_half;
	int base;
	int ht;
    int idxend;
};

__constant__ discConstants disc;

// Constants for register version.  Ends with R to distinguish.
#define BASER            36
#define HEIGHTR          18
#define WARPSIZER        32
#define TWOBASER         72

//Initial condition.
__host__ 
REAL initFun(REAL xnode)
{
	return TWO * cos(19.0*xnode*M_PI/128.0);
}