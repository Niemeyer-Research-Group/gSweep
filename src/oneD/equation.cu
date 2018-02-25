/**
    The general equation functions.
*/

include "equation.h"

Equation::Equation(inputf inFile, str outpath, int argc=0, char *argv[]="");
{
    inFile >> inJ;
    inFile.close();
    parseArgs(argc, argv);
    chosenEquation = new *Specific();

    stateSize = sizeof(states);
    gridSize = inJ["nX"].asInt();
    lx = inJ["lx"].asDouble();
    bitSize = stateSize*gridSize;
    freq = inJ['freq'].asDouble();
    tpb = inJ['tpb'].asInt();
    tf = inJ['tf'].asDouble();
    freq = inJ['freq'].asDouble();
    dx = lx/(double)gridSize; // Spatial step
    bks = gridSize/tpb;

    if (!freq) freq = tf*2.0;

    rdir = outpath;
    tpath = rdir + "/t" + fspec;
    spath = rdir + "/s" + fspec;

    nWrite = tf/freq + 2; //Future, use to preallocate solution array. (OR USE vector)
    inJ["dx"] = dx;
    chosenEquation->initEq(inJ);

};

void Equation::makeInitialContidion(states *nState)
{
    chosenEquation->initState(nState, idx)
}

Equation::solutionOutput(states *outState, double tstamp)
{
    str tsts = std::to_string(tstamp);
    for (int k=0; k<NVARS; k++)
    {
        for (int i=0; i<gridSize; i++)
        {
            str xpts = std::to_string((double)i*dx);
            solution[outVars[k]][tsts][xpts] = chosenEquation->printout(outState + i, k);
        }
    }
}



    




