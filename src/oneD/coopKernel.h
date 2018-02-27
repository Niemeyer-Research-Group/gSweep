/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/


#include "equation.h"
#include "kernel.h"
#include <cooperative_groups.h>

using namespace cooperative_groups;

__global__ 
void classicKernel(states *state, int tstart, int tend)
{
    grid_group grid = this_grid();
    int idxes[3];

    for (int k = tstart; k<tend; k++)
    {
        for(int i = grid.thread_rank(); i < deqConstants.nX; i += grid.size())
        {
            stepUpdate(state, idxes, k);
        }
    }
    grid.sync()
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

__device__
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

__device__
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

__device__
void wholeDiamond(states *statein, int tstep)
{
    // THE DEBATE IS WHETHER OR NOT TO PUT THE SUBTIMESTEP COUNT IN THE ARRAY.
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
}

__global__
void sweptKernel(states *statein, states *stateout, int tstep)
{
    extern __shared__ states tState[];
    for (int k = tstart; k<tend; k++)
    {
        sweepRead(tState, statein, gid, tid, blockDim.x);
        for(int i = grid.thread_rank(); i < deqConstants.nX; i += grid.size())
        {
            stepUpdate(state, idxes, k);
        }
    }
}

int getLaunch(void *func)
{
    int nbksm;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbksm, func, tpb, 0));

    return deviceProp.multiProcessorCount * nbksm;
}

void Solver::classicCooperative()
{

    void *kernelFunc = (void *) classicKernel;

    bks = getLaunch(kernelFunc);

    eq.spath += "_Classic_Cooperative.json";
    eq.tpath += "_Classic_Cooperative.csv";
    std::cout << "Classic scheme" << std::endl;
    void *args[] = {(void*)&dState, (void*) &tstep};

    cudaLaunchCooperativeKernel(kernelFunc, bks, tpb, args, NULL, NULL);
}

void Solver::sweptCooperative()
{
    eq.spath += "_Swept_Cooperative.json";
    eq.tpath += "_Swept_Cooperative.csv";
    std::cout << "Swept scheme" << std::endl;
    states *dState2;

    void *kernelFunc = (void *) classicKernel;
    bks = getLaunch(kernelFunc);

    cudaMalloc((void **)&dState2, bitAlloc);
    void **args[3] = {dState, dState2, &tstep};
    
    cudaLaunchCooperativeKernel(kernelFunc, bks, tpb, args, smem, NULL);

    cudaFree(dState2);
}
