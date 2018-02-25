/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

#include "kernel.h"

__global__ 
void classicStep(states *state, int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int idxes[3];
    
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


void Solver::classic()
{
    cout << "Classic scheme" << endl;

    while (t_eq <= eq.tf)
    {
        classicStep <<< eq.bks, eq.tpb >>> (dks_in, tstep);
        t_eq += eq.dt;
        tstep++;

        if (t_eq > twrite)
        {
            this->storeSolution();
            twrite += eq.freq;
        }
    }
}

void Solver::swept()
{
    std::cout << "Swept scheme" << std::endl;

    // WHy 2?
    cudaMalloc((void **)&stateA, tBytes);
    cudaMalloc((void **)&stateB, tBytes);

    //inline dir = -1, split dir = 1 because passing after calculation.
    upTriangle <<< eq.bks, eq.tpb, smem >>> (stateA, stateB, tstep);
    wholeDiamond <<< eq.bks, eq.tpb, smem >>> (stateB, stateA, tstep, 1);

    while (t_eq <= eq.tf)
    {
        wholeDiamond <<< eq.bks, eq.tpb, smem >>> (stateA, stateB, tstep, -1);
        t_eq += eq.dt;
        tstep++;

        if (t_eq > twrite)
        {
            downTriangle <<< eq.bks, eq.tpb, smem >>> (stateB, stateA, tstep);
            this->storeSolution();
            twrite += eq.freq;
        }
    }   
}

void Solver::solveEquation()
{
    if ()

}

//Must be accessible from swept and classic.
void Solver::storeSolution()
{
    cudaMemcpy(hState, dState, eq.bitsize, cudaMemcpyDeviceToHost);
    eq.solutionOutput(hState, t_eq);
}

void Solver::writeFiles(double timed)
{
    this->storeSolution();
    double per_ts = timed/(double) tstep;
    timeOut = fopen(tpath.c_str(), "a+");
    fseek(timeOut, 0, SEEK_END);
    int ft = ftell(timeOut);
    if (!ft) fprintf(timeOut, "tpb,nX,time\n");
    fprintf(timeOut, "%d,%d,%.8f\n", tpb, gridSize, per_ts;
    fclose(timeOut);
}