include "equation.h"

Equation::Equation(inputf inFile, char* outpath, int argc=0, char *argv[]="")
{
    inFile >> inJ;
    parseArgs(argc, argv);
    initializeGrid():
}

Equation::solutionOutput(states *outState, double tstamp, int idx, int strt)
{
    std::string tsts = std::to_string(tstamp);
    double xpt = indexer(cGlob.dx, idx, strt);
    std::string xpts = std::to_string(xpt);
    for (int k=0; k<NVARS; k++)
    {
        solution[outVars[k]][tsts][xpts] = printout(outState + idx, k);
    }
}


Equation::stepUpdate(states *state, int idx, int ins)
{
    return answer
}