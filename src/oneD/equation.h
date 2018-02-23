/*
    Base class for equations
*/
#ifndef EQ_H
#define EQ_H

#include "equations/eqHead.h"

class Equation
{
private:
	jsons inJ, solution;

	void parseArgs(int argc, char *argv[])
	{
		if (argc>4)
		{
			str inarg;
			for (int k=4; k<argc; k+=2)
			{
				inarg = argv[k];
				inJ[inarg] = atof(argv[k+1]);
			}
		}
	}

public:
	REAL dt, lx, tf, freq;
	int tpb, gridSize, stateSize, bitSize;
	const REAL pi = M_PI;
	char sout, tout;
	double dx;
	int bks;
	outputf timingOut, solutionOut;

	Equation(inputf inFile, char* outpath, int argc=0, char *argv[]="")
	{
		inFile >> inJ;
		inFile.close();
    	parseArgs(argc, argv);

		stateSize = sizeof(states);
		gridSize = inJ["nX"].asInt();
		freq = inJ['freq'].asDouble();
		tpb = inJ['tpb'].asInt();
		tf = inJ['tf'].asDouble();
		//bitSize = ;
		specificInit(this);
	};

	void solutionOutput(states *outState, double tstamp, int idx, int strt);

	// The uninitialized but guaranteed functions

	REAL printout(states *state, int i);

	void writeSolution();

	void writeTime();
};

#endif
