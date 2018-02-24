#ifndef KERNELMAIN_H
#define KERNELMAIN_H

#include "equation.h"

class Solver
{
private:
    FILE *timeOut;
    int tstep;
    states *hState, *dState;
    str kernelType, schemeType;
    Equation eq;

    void classic();
    void swept();

public:
    //Some members
    Solver(const Equation &eq, str scheme, str kernel)
    {
        kernelType = kernel;
        schemeType = scheme;

        cudaSetDevice(GPUNUM);
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

        cudaHostAlloc(hState, eq.bitSize + 2*eq.stateSize)
        cudaMalloc(dState, eq.bitSize + 2*eq.stateSize)
        eq.makeInitialCondition(hState);
        cudaMemcpyToSymbol(deqConstants, &heqConstants, sizeof(equationConstants));
        cudaMemcpy(dState, hState, eq.bitsize, cudaMemcpyHostToDevice);
    }

    void solveEquation();

    void writeOut();

    ~Solver::Solver()
    {
        //WriteOut?
        cudaFreeHost(hState);
        cudaFree(dState);
        cudaDeviceSynchronize();
    }

};

#endif