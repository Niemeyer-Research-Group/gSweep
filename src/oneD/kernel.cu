/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

#include "kernel.h"
//#include "coopKernel.h"

__global__ 
void classicStep(states *state, int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int idxes[3];
    
    stepUpdate(state, idxes, ts);
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
    int gidout = (gid - mid) % deqConstants.lastIndex;
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
    int gidout = (gid + dir*mid) % deqConstants.lastIndex;
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
    eq->spath += "_Classic_Normal.json";
    eq->tpath += "_Classic_Normal.csv";
    std::cout << "Classic scheme" << std::endl;

    while (t_eq <= eq->tf)
    {
        classicStep <<< eq->bks, eq->tpb >>> (dState, tstep);
    
        tstep++;
        t_eq += (eq->dt * tstep/NSTEPS);

        if (t_eq > twrite)
        {
            this->storeSolution();
            twrite += eq->freq;
        }
    }
}

void Solver::swept()
{
    eq->spath += "_Swept_Normal.json";
    eq->tpath += "_Swept_Normal.csv";
    std::cout << "Swept scheme" << std::endl;

    states *dState2;

    cudaMalloc((void **)&dState2, bitAlloc);

    //inline dir = -1, split dir = 1 because passing after calculation.
    upTriangle <<< eq->bks, eq->tpb, smem >>> (dState, dState2, tstep);
    wholeDiamond <<< eq->bks, eq->tpb, smem >>> (dState2, dState, tstep, 1);

    while (t_eq <= eq->tf)
    {
        wholeDiamond <<< eq->bks, eq->tpb, smem >>> (dState, dState2, tstep, -1);
        tstep += eq->height;
        t_eq += (eq->dt * tstep/NSTEPS);

        if (t_eq > twrite)
        {
            downTriangle <<< eq->bks, eq->tpb, smem >>> (dState2, dState, tstep);
            this->storeSolution();
            twrite += eq->freq;
        }
    }   
    cudaFree(dState2);
}

void Solver::solveEquation()
{
    cudaTime timer;
    timer.tinit();
    if (!schemeType.compare("S"))
    {
        this->swept();
    }
    else if (!schemeType.compare("C"))
    {
        this->classic();
    }
    else
    {
        std::cerr << "Incorrect or no scheme given! " << std::endl;
    }
    timer.tfinal();
    timed = timer.getLastTime();
}

//Must be accessible from swept and classic.
void Solver::storeSolution()
{
    cudaMemcpy(hState, dState, eq->bitSize, cudaMemcpyDeviceToHost);
    eq->solutionOutput(hState, t_eq);
}

void Solver::writeFiles()
{
    this->storeSolution();
    double per_ts = timed/(double) tstep;
    timeOut = fopen(eq->tpath.c_str(), "a+");
    fseek(timeOut, 0, SEEK_END);
    int ft = ftell(timeOut);
    if (!ft) fprintf(timeOut, "tpb,nX,time\n");
    fprintf(timeOut, "%d,%d,%.8f\n", eq->tpb, eq->gridSize, per_ts);
    fclose(timeOut);
}