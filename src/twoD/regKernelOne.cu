/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

#include "waveConsts.h"
#include "mainGlobals.h"

// Leapfrog!
__device__  
void stepUpdate(states *state, int idx[3], int tstep)
{
    int ins = tstep&1; // In step with
    stencil(state, idx, ins);
}


__device__ void boundary(int gid, int tid, int *idxes)
{
    idxes[1] = tid;
    if (deqConsts.typ == 0)
    {
        idxes[0] = (gid) ? tid : deqConsts.idxF;
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
    boundary(gid, gid, &idxes[0]);
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
            tState[0] = statein[deqConsts.idxF];
        }   
        else if (gid == deqConsts.idxF)
        {
            tState[blockDim.x + 1] = statein[0];
        }
        else
        {
            tState[tadj] = statein[(gid-1) + tadj];
        }
    }
}

__global__
void upTriangle(states *statein, states *stateout, int tstep)
{   
    extern __shared__ states tState[];

    //Global Thread ID
    int tid = threadIdx.x; // Thread index
    int gid = blockDim.x * blockIdx.x + threadIdx.x; 
    int tidx = threadIdx.x; //Block Thread ID
    int mid = blockDim.x >> 1;
    int gidout = (gid - mid) % deqConsts.idxF;
    int tnow = tstep;
    int idxes[3];
    for (int k=-1; k<2; k++) idxes[k+1] = tid + k;

    tState[tidx] = statein[gid];

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
                stepUpdate(tState, idxes, tnow);
        }
        tnow++;
        __syncthreads();
    }
    stateout[gid] = tState[tidx];
}

__global__
void wholeDiamond(states *statein, states *stateout, int tstep, int dir)
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
        	stepUpdate(tState, idxes, tnow);
		}
		tnow++;
		__syncthreads();
	}

	for (int k=2; k<=mid; k++)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepUpdate(tState, idxes, tnow);
		}
		tnow++;
		__syncthreads();
    }
    stateout[gidout] = tState[tidx];
}

double classicWrapper(states *state, int *tstep)
{
    cout << "Classic scheme" << endl;
    states *dks_in;
    int tmine = *tstep;
    int tBytes = cGlob.szState*cGlob.nX;
    cudaMalloc((void **)&dks_in, tBytes);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dks_in, state, tBytes, cudaMemcpyHostToDevice);

    double t_eq = NSTEPS * cGlob.dt;
    double twrite = cGlob.freq - 0.25*cGlob.dt;

    while (t_eq <= cGlob.tf)
    {
        classicStep <<< cGlob.bks, cGlob.tpb >>> (dks_in, tmine);
        t_eq += cGlob.dt;
        tmine++;

        if (t_eq > twrite)
        {
            cudaMemcpy(state, dks_in, tBytes, cudaMemcpyDeviceToHost);
            twrite += cGlob.freq;
        }
    }
    cudaMemcpy(state, dks_in, tBytes, cudaMemcpyDeviceToHost);
    cudaFree(dks_in);
    return t_eq;
}

double sweptWrapper(states *state, int *tstep)
{
    cout << "Swept scheme" << endl;
    states *stateA, *stateB;
    int tmine = *tstep;
    const int tBytes = cGlob.szState*cGlob.nX;
    const size_t smem = (cGlob.tpb + 2) * cGlob.szState;

    cudaMalloc((void **)&stateA, tBytes);
    cudaMalloc((void **)&stateB, tBytes);

    // Copy the initial conditions to the device array.
    cudaMemcpy(stateA, state, tBytes, cudaMemcpyHostToDevice);

    double t_eq = NSTEPS * cGlob.dt;
    double twrite = cGlob.freq - 0.25*cGlob.dt;
    //inline dir = -1, split dir = 1 because passing after calculation.
    upTriangle <<< cGlob.bks, cGlob.tpb, smem >>> (stateA, stateB, tmine);
    wholeDiamond <<< cGlob.bks, cGlob.tpb, smem >>> (stateB, stateA, tmine, 1);

    while (t_eq <= cGlob.tf)
    {
        wholeDiamond <<< cGlob.bks, cGlob.tpb, smem >>> (stateA, stateB, tmine, -1);
        t_eq += cGlob.dt;
        tmine++;

        if (t_eq > twrite)
        {
            downTriangle <<< cGlob.bks, cGlob.tpb, smem >>> (stateB, stateA, tmine);
            cudaMemcpy(state, stateA, tBytes, cudaMemcpyDeviceToHost);
            twrite += cGlob.freq;
        }
    }
    cudaMemcpy(state, stateA, tBytes, cudaMemcpyDeviceToHost);
    cudaFree(stateA);
    cudaFree(stateB);
    return t_eq;
}
