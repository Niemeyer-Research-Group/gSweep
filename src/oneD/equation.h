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

// UTILITIES PATH
#include "myVectorTypes.h"
#include "json/json.h"
#include "cudaUtils.h"

#define REAL        double
#define ZERO        0.0
#define QUARTER     0.25
#define HALF        0.5
#define ONE         1.0
#define TWO         2.0
// Initial step, normally 1 but here it's 2 bc two timesteps are exact and initial. (0, dt)

typedef Json::Value jsons;
typedef std::istream inputf;
typedef std::ostream outputf;

struct Equation
{
private:
	REAL dt, dx, lx, tf, freq;
	int tpb, bks, gridSize, stateSize, bitSize;
	const REAL pi = M_PI;
	char sout, tout;

	jsons inJ, solution;
	outputf timingOut, solutionOut;

	void initializeGrid()
	{
		freq = inJ['freq'];
		tpb = inJ['tpb'];
		tf = inJ['tf'];
	}

	void initializeEquation(); // The specific constructor (needs to be called AFTER this constructor... HOW?)

	void parseArgs(int argc, char *argv[])
	{
		if (argc>4)
		{
			std::string inarg;
			for (int k=4; k<argc; k+=2)
			{
				inarg = argv[k];
				inJ[inarg] = atof(argv[k+1]);
			}
		}
	}


public:

	Equation(inputf inFile, char* outpath, int argc=0, char *argv[]="");

	void solutionOutput(states *outState, double tstamp, int idx, int strt);

	// The uninitialized but guaranteed functions

	REAL printout(states *state, int i);

	__device__ __host__ void stepUpdate(states *state, int idx, int ins);
    
};

#endif
