/**
    Base class for equations
*/

#ifndef EQ_H
#define EQ_H

#include "equations/eqHead.h"

class Equation
{
private:
	jsons inJ, solution;
	Specific *chosenEquation;

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
	str rdir
	str tpath, spath;
	double dt, lx, tf, freq;
	int tpb, gridSize, stateSize, bitSize;
	char sout, tout;
	double dx;
	int bks;
	outputf solutionOut;

	Equation(){};

	Equation(inputf inFile, char* outpath, int argc=0, char *argv[]="")
	{
		inFile >> inJ;
		inFile.close();
    	parseArgs(argc, argv);
		chosenEquation = new *Specific();

		stateSize = sizeof(states);
		gridSize = inJ["nX"].asInt();
		freq = inJ['freq'].asDouble();
		tpb = inJ['tpb'].asInt();
		tf = inJ['tf'].asDouble();

		tpath = rdir + "/t" + fspec + scheme + t_ext;
		bitSize = stateSize*gridSize;

		chosenEquation->initEq(inJ);

	};

	void makeInitialContidion(states *nState, int k);

	void solutionOutput(states *outState, double tstamp, int k);

	// The uninitialized but guaranteed functions

	void writeSolution();

	void writeTime();

	~Equation()
	{
		delete[] chosenEquation;	
	};
};

#endif
