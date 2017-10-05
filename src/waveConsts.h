/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

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

#include "waveConsts.h"
#include "myVectorTypes.h"
#include "json/json.h"

#define GPUNUM          0

#define REAL        double
#define ZERO        0.0
#define QUARTER     0.25
#define HALF        0.5
#define ONE         1.0
#define TWO         2.0

typedef Json::Value jsons;

using namespace std;

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

globalism cGlob;

void equationSpecificArgs(jsons inJs);
{
	cGlob.c = inJs["c"].asDouble();
	cGlob.w = inJs["w"].asDouble();
	cGlob.dt = inJs["dt"].asDouble();
	cGlob.cfl = inJs["cfl"].asDouble();
	if (cGlob.cfl>0) 
	{
		cGlob.dt = cGlob.cfl*cGlob.dx/cGlob.c;
	}
	else if(cGlob.dt>0)
	{
		cGlob.cfl = cGlob.c*cGlob.dt/cGlob.dx;
	}
	else
	{
		cGlob.cfl = 0.5;
		cGlob.dt = cGlob.cfl*cGlob.dx/cGlob.c;
	}
	heqConsts.cFLsq = cGlob.cfl * cGlob.cfl;
	heqConsts.idxF = cGlob.nX-1;
	heqConsts.typ = 0;
}