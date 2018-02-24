/**
    The general equation functions.
*/

include "equation.h"

void Equation::makeInitialContidion(states *nState)
{
    chosenEquation->initState(nState, idx)
}

Equation::solutionOutput(states *outState, double tstamp, int idx)
{
    str tsts = std::to_string(tstamp);
    str xpts = std::to_string(xpt);
    for (int k=0; k<NVARS; k++)
    {
        solution[outVars[k]][tsts][xpts] = chosenEquation->printout(outState + idx, k);
    }
}


Equation::writeSolution()
{

}

Equation::writeTime()
{
    // Write out performance data as csv
    
    timeOut = fopen(tpath.c_str(), "a+");
    fseek(timeOut, 0, SEEK_END);
    int ft = ftell(timeOut);
    if (!ft) fprintf(timeOut, "tpb,gpuA,nX,time\n");
    fprintf(timeOut, "%d,%.4f,%d,%.8f\n", cGlob.tpb, cGlob.gpuA, cGlob.nX, per_ts);
    fclose(timeOut);
}



