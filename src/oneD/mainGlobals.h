#ifndef MGLOB_H
#define MGLOB_H

// Equation, grid, affinity data
void readIn(int argc, char *argv[])
{ 
	ifstream injson(argv[2], ifstream::in);
	injson >> inJ;
	injson.close();
	parseArgs(argc, argv);
}

//DEFINITIONS OF struct would need to be merged from two different places.
__host__ 
void initConsts()
{
    cGlob.szState = sizeof(states);
    cGlob.tpb = inJ["tpb"].asInt();
    cGlob.nX = inJ["nX"].asInt();
    cGlob.bks = cGlob.nX/cGlob.tpb;

    cGlob.lx = inJ["lx"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();

    cGlob.dx = cGlob.lx/((double)cGlob.nX);
    if (!cGlob.freq) cGlob.freq = cGlob.tf*2.0;
    equationSpecificArgs(inJ);
}

#endif
