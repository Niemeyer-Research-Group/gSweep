/*
    Base class for equations
*/
#ifndef EQ_H
#define EQ_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <iostream>
#include <fstream>
#include <ostream>
#include <istream>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "myVectorTypes.h"
#include "json/json.h"

#define REAL        double
#define ZERO        0.0
#define QUARTER     0.25
#define HALF        0.5
#define ONE         1.0
#define TWO         2.0
#define TSTEPI		2	// Initial step, normally 1 but here it's 2 bc two timesteps are exact and initial. (0, dt)

struct eqConsts{
	REAL cFLsq;
	int idxF, typ; //Type is Boundary condition type. 0 for periodic.
};

__constant__ eqConsts deqConsts;
eqConsts heqConsts;

struct states{
    REAL u[2];
};

// Damn. How should I encapsulate this.  Perhaps I should make a class.
struct globalism{
	REAL dt, dx, c, w, lx, tf, freq, cfl;
	int tpb, bks, nX, szState;
	const REAL pi = M_PI;
};

class equation
{

private:
    

public:

    equation();

    __device__ __host__ void stepUpdate();
    
};

#endif
