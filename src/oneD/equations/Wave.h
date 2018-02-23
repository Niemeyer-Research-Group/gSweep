/** 
    Copyright (C) 2018 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/
#ifndef WAVE_H
#define WAVE_H

#include "rootHeader.h"

#define NSTEPS		2
#define TSTEPI		2	
#define 

struct states{
    REAL u[2];
};

struct equationConstants{
	REAL cflSQUARED;
	int lastIndex;	
};

__constant__ equationConstants deqConstants;
equationConstants heqConstants;

namespace Wave
{
	double c, w, cfl;
}

void specificInit(Equation &e)
{	 
	using namespace Wave;
	c = e.inJ["c"].asDouble();
	w = e.inJs["w"].asDouble();
	e.dt = e.inJs["dt"].asDouble();
	cfl = e.inJs["cfl"].asDouble();

	if (cfl>0) 
	{
		e.dt = cfl*e.dx/c;
	}
	else if(dt>0)
	{
		cfl = c*e.dt/e.dx;
	}
	else
	{
		cfl = 0.5;
		e.dt = cfl*e.dx/c;
	}

	heqConstants.cflSQUARED = cfl * cfl;
	heqConstants.lastIndex = e.gridSize-1;
	//heqConstants.typ = 0;
}

void initializeState(Equation &e, states *state, int n, int k)
{
	using namespace Wave;
	state[n].u[k] = cos(e.dt*w*e.pi*k) * sin(e.dx*w*e.pi*n);
}

// Leapfrog. 
__device__  __host__
void stepUpdate(states *state, int idx, int ins)
{
    int offs = ins^1;
    state[idx[1]].u[ins] = TWO * state[idx[1]].u[offs] * (1 - deqConsts.cFLsq) + deqConsts.cFLsq * (state[idx[0]].u[offs] + state[idx[2]].u[offs] - state[idx[1]].u[ins]);
}

// REAL errorNorm(states *state, REAL t)
// {
//     REAL *stateE = malloc(sizeof(REAL) * .nX); //??
//     REAL stateF = 0.0;
//     REAL sq;
//     int sw = t&1; // Not t itself, the n timesteps
//     for (int n=0; n<.nX; n++) 
//     {
//         exactSolution(stateE, n, t); // t is tsteps
//         sq = std::abs(stateE[n] - state[n].u[sw]);
//         stateF += sq*sq;
//     }
//     sq = std::sqrt(stateF);
//     std::cout << "The RMSE is: " << sq << std::endl;
//     return sq; // THe RMSE!
// }

#endif