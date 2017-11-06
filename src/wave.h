/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/
#ifndef WAVE_H
#define WAVE_H

#define TSTEPI		2	// Initial step, normally 1 but here it's 2 bc two timesteps are exact and initial. (0, dt)

typedef Json::Value jsons;

using namespace std;

class Wave
{

	private: 

	globalism cGlob;

	// Initial conditions.  Could be macro.
	__host__ void exactSolution(states *state, int n, int k);

	// Leapfrog. 
	__device__ void stencil(states *state, int idx[3], int ins);

	__host__ void equationSpecificArgs(jsons inJs);

	__host__ REAL errorNorm(states *state, REAL t);

	__device__ void stencil(states *state, int idx[3], int ins);

}

#endif