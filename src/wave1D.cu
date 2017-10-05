/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

// Fuckin duh.  You can't do it like this you need an array!

#include "waveConsts.h"

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
    state[idx[1]]->u[ins] = TWO * state[idx[1]]->u[offs] * (1 - deqConsts.cfl) + deqConsts.cfl * (state[idx[0]]->u[offs] + state[idx+[2]]->u[offs]) - state[idx[1]]->u[ins];
}

// Leapfrog!
__device__  
void stepUpdate(states *state, int idx[3], int tstep)
{
    int ins = tstep&1; // In step with
    stencil(state, idx, ins);
}

__host__ 
void initConsts(jsons inJ)
{
    cGlob.szState = sizeof(states);
    cGlob.tpb = inJ["tpb"].asInt();
    cGlob.nX = inJ["nX"].asInt();
    cGlob.bks = cGlob.nX/cGlob.tpb;

    cGlob.lx = inJ["lx"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();

    cGlob.dx = cGlob.lx/((double)cGlob.nX);
    if (!cGlob.freq) cGlob.freq = cGlob.tf*2.0;
    equationSpecificArgs(inJ);
}

__device__ void boundary(int gid, int tid, int *idxes)
{
    idxes[1] = tid;
    if (deqConsts.typ == 0)
    {
        idxes[0] = (gid) ? tid : deqconsts.idxF;
        idxes[2] = (gid == deqConsts.idxF) ? 0 : tid;   
    }
    // // Reflective
    // else
    // {
    //     if (gid == 1) state[0] = state[2];
    //     if (gid = deqConsts.idxF-1) state[deqConsts.idxF] = state[deqConsts.idxF-2]; 
    // }
}

__global__ 
void classicStep(states *state, int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int idxes[3];
    boundary(gid, gid, &idxes);
    stepUpdate(state, idxes, ts);
}

__device__
__forceinline__
void sweepRead(states *tState, states *statein, int gid, int tid, int bd)
{
    int tadj = tid * (bd + 1);
    tState[tid+1] = statein[gid];
    __syncthreads();
    if (tid<2)
    {
        if (gid == 0)
        {
            tState[0] = statesin[deqConsts.idxF];
        }   
        else if (gid == deqConsts.idxF)
        {
            tState[blockDim.x + 1] = statesin[0];
        }
        else
        {
            tState[tadj] = statesin[(gid-1) + tadj];
        }
    }
}

__global__
void upTriangle(states *statein, states *stateout, int tstep)
{   
    extern __shared__ states tState[];

    //Global Thread ID
    int gid = blockDim.x * blockIdx.x + threadIdx.x; 
    int tidx = threadIdx.x; //Block Thread ID
    int mid = blockDim.x >> 1;
    int gidout = (gid - mid) % deqConsts.idxF;
    int tnow = tstep;
    int idxes[3];
    for (int k=-1; k<2; k++) idxes[k+1] = tid + k;

    tState[tidx] = statesin[gid];

    __syncthreads();

    for (int k=1; k<mid; k++)
    {
        if (tidx < (blockDim.x-k) && tidx >= k)
        {
            stepUpdate(tState, idxes, tnow); 
        }
        tnow++;
        __syncthreads();
    }
    stateout[gidout] = tState[tidx];
}

__global__
void downTriangle(states *statein, states *stateout, int tstep)
{
    extern __shared__ states tState[];

    int tid = threadIdx.x; // Thread index
    int mid = blockDim.x >> 1; // Half of block size
    int base = blockDim.x + 2; 
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = tid + 1;
    int tnow = tstep; // read tstep into register.
    int idxes[3];
    for (int k=-1; k<2; k++) idxes[k+1] = tidx + k;

    sweepRead(tState, statein, gid, tid, blockDim.x);
    
    for (int k=mid; k>0; k--)
    {
        if (tidx < (base-k) && tidx >= k)
        {
                stepUpdate(temper, idxes, tnow);
        }
        tnow++;
        __syncthreads();
    }
    state[gid] = temper[tidx];
}

__global__
void wholeTriangle(states *statein, states *stateout, int tstep, int dir)
{
    extern __shared__ states tState[];

    int tid = threadIdx.x; // Thread index
    int mid = blockDim.x >> 1; // Half of block size
    int base = blockDim.x + 2; 
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int gidout = (gid + dir*mid) % deqConsts.idxF;
    int tidx = tid + 1;
    int tnow = tstep; // read tstep into register.
    int idxes[3];
    for (int k=-1; k<2; k++) idxes[k+1] = tidx + k;

    sweepRead(tState, statein, gid, tid, blockDim.x);

    for (int k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
        	stepUpdate(temper, tidx, tnow);
		}
		tnow++;
		__syncthreads();
	}

	for (int k=2; k<=mid; k++)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepUpdate(temper, tidx, tnow);
		}
		tnow++;
		__syncthreads();
    }
    
    state[gidout] = temper[tidx];
}

void classicWrapper()
{

}

void sweptWrapper()
{
    //inline dir = -1, split dir = 1 because passing after calculation.
}

void errorNorm(states *state, REAL t, REAL w)
{

}