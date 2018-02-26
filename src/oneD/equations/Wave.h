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

/**
	The beginning of the user defined scheme
*/
#define NVARS		1
#define NSTEPS		2
#define TSTEPI		2	

struct states{
    REAL u[2];
};

struct equationConstants{
	REAL cflSQUARED;
	int lastIndex;	
};
const str outVars[NVARS] = {"VELOCITY"}; 
const str fspec = "Wave";

__constant__ equationConstants deqConstants;
equationConstants heqConstants;

struct Wave
{
	const double pi = M_PI;
	double c, w, dt, dx;
	double cfl;

	void initEq(jsons inJs)
	{	 
		c = inJs["c"].asDouble();
		w = inJs["w"].asDouble();
		dt = inJs["dt"].asDouble();
		cfl = inJs["cfl"].asDouble();
		dx = inJs["dx"].asDouble();
		nX = inJs["nX"].asDouble();

		if (!cfl) 
		{
			cfl=c*dt/dx;;
		}
		else
		{
			c = cfl*dx/dt;
		}

		heqConstants.cflSQUARED = cfl * cfl;
		heqConstants.lastIndex = nX-1;
	}

	static void initState(states *state, int n)
	{
		for (int k=0; k<2; k++) state[n].u[k] = cos(dt * w * pi * k) * sin(dx * w * pi * n);
	}

	double printout(states * state, int i)
	{
		return state->u[0];
	}
}

typedef Wave Specific;

// OTHER ELEMENTS OF THE SCHEME GO HERE

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