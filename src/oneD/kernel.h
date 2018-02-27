#ifndef KERNELMAIN_H
#define KERNELMAIN_H

#include "equation.h"
#include "cudaUtils.h"

class Solver
{
private:
    Equation *eq;
    FILE *timeOut;
    int tstep, bitAlloc;
    size_t smem;
    states *hState, *dState;
    str kernelType, schemeType;
    double t_eq, twrite, timed;

    void storeSolution();
    void classic();
    void swept();

public:
    //Some members
    Solver(Equation *eqparam, str scheme, str kernel)
    {
        //I'd Prefer to work with a pointer.
        eq = eqparam; //Requires a copy constructor.
        kernelType = kernel;
        schemeType = scheme;
        tstep = TSTEPI;
        bitAlloc = eq->bitSize + 2*eq->stateSize;
        t_eq = NSTEPS * eq->dt;
        twrite = eq->freq - 0.25*eq->dt;
        smem = (eq->tpb + 2) * eq->stateSize;
        

        cudaHostAlloc((void **) &hState, bitAlloc, cudaHostAllocDefault);
        cudaMalloc((void **) &dState, bitAlloc);
        eq->makeInitialCondition(hState);
        cudaMemcpyToSymbol(deqConstants, &heqConstants, sizeof(equationConstants));
        cudaMemcpy(dState, hState, bitAlloc, cudaMemcpyHostToDevice);
    }    
    ~Solver()
    {
        cudaFreeHost(hState);
        cudaFree(dState);
        cudaDeviceSynchronize();
    }

    void solveEquation();

    void writeFiles();

};

#endif