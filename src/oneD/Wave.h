/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

#include "equation.h"

#define NSTEPS		2
#define TSTEPI		2	
using namespace std;

// These structs can be declared in abc, but we don't know what they are.

struct eqConsts{
	REAL cFLsq;
	int idxF;
};

__constant__ eqConsts deqConsts;
eqConsts heqConsts;

struct states{
    REAL u[2];
};

class Wave: class Equation
{
private:
	c, w, cfl;

public:
	//??
	Wave(jsons inJs)
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

__host__ 
void exactSolution(states *state, int n, int k)
{
	 state[n].u[k] = cos(cGlob.dt*cGlob.w*cGlob.pi*k) * sin(cGlob.dx*cGlob.w*cGlob.pi*n);
}

// Leapfrog. 
__device__  __host__
void stepUpdate(states *state, int idx, int ins)
{
    int offs = ins^1;
    state[idx[1]].u[ins] = TWO * state[idx[1]].u[offs] * (1 - deqConsts.cFLsq) + deqConsts.cFLsq * (state[idx[0]].u[offs] + state[idx[2]].u[offs] - state[idx[1]].u[ins]);
}

REAL errorNorm(states *state, REAL t)
{
    REAL *stateE = malloc(sizeof(REAL) * cGlob.nX); //??
    REAL stateF = 0.0;
    REAL sq;
    int sw = t&1; // Not t itself, the n timesteps
    for (int n=0; n<cGlob.nX; n++) 
    {
        exactSolution(stateE, n, t); // t is tsteps
        sq = std::abs(stateE[n] - state[n].u[sw]);
        stateF += sq*sq;
    }
    sq = std::sqrt(stateF);
    std::cout << "The RMSE is: " << sq << std::endl;
    return sq; // THe RMSE!
}
}