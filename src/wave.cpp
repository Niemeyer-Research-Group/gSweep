/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

#define TSTEPI		2	// Initial step, normally 1 but here it's 2 bc two timesteps are exact and initial. (0, dt)

typedef Json::Value jsons;

using namespace std;


// Initial conditions.  Could be macro.
__host__ 
void exactSolution(states *state, int n, int k)
{
	 state[n]->u[k] = cos(cGlob.dt*cGlob.w*cGlob.pi*k) * sin(cGlob.dx*cGlob.w*cGlob.pi*n);
}

// Leapfrog. 
__device__
void stencil(states *state, int idx[3], int ins)
{
    int offs = ins^1;
    state[idx[1]]->u[ins] = TWO * state[idx[1]]->u[offs] * (1 - deqConsts.cFLsq) + deqConsts.cFLsq * (state[idx[0]]->u[offs] + state[idx+[2]]->u[offs]) - state[idx[1]]->u[ins];
}

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

REAL errorNorm(states *state, REAL t)
{
    REAL *stateE = malloc(sizeof(REAL) * cGlob.nX);
    REAL stateF = 0.0;
    REAL sq;
    int sw = t&1;
    for (int n=0; n<nX; n++) 
    {
        exactSolution(stateE, n, t); // t is tsteps
        sq = std::abs(stateE[n] - state[n].u[sw]);
        stateF += sq*sq;
    }
    sq = std::sqrt(stateF);
    std::cout << "The RMSE is: " << sq << std::endl;
    return sq; // THe RMSE!
}