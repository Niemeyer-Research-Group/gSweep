#ifndef KERNELMAIN_H
#define KERNELMAIN_H

#include "equation.h"

class Solver
{
private:
    Equation *eq;
    FILE *timeOut;
    int tstep, bitAlloc;
    size_t smem;
    states *hState, *dState;
    str kernelType, schemeType;
    double t_eq, twrite;

    void storeSolution();
    void classic();
    void swept();

public:
    //Some members
    Solver(const Equation *eqparam, str scheme, str kernel)
    {
        //I'd Prefer to work with a pointer.
        eq = eqparam; //Requires a copy constructor.
        kernelType = kernel;
        schemeType = scheme;
        tstep = TSTEPI;
        bitAlloc = eq.bitSize + 2*eq.stateSize;
        t_eq = NSTEPS * eq.dt;
        twrite = eq.freq - 0.25*eq.dt;
        smem = (eq.tpb + 2) * eq.stateSize;

        cudaSetDevice(GPUNUM);
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

        cudaHostAlloc((void **) &hstate, bitAlloc, cudaHostAllocDefault);
        cudaMalloc((void **) &dstate, bitAlloc);
        eq.makeInitialCondition(hState);
        cudaMemcpyToSymbol(deqConstants, &heqConstants, sizeof(equationConstants));
        cudaMemcpy(dState, hState, bitAlloc, cudaMemcpyHostToDevice);
        this->storeSolution();
    }    
    ~Solver::Solver()
    {
        cudaFreeHost(hState);
        cudaFree(dState);
        cudaDeviceSynchronize();
    }

    void solveEquation();

    void writeFiles(double timed);

};

#endif