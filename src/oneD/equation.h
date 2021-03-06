/**
    Base class for equations
*/

#ifndef EQ_H
#define EQ_H

#include "equations/eqHead.h"

//str defaultOut = "/something/From/Makefile";

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
	str tpath, spath;
	double dt, lx, tf, freq;
	int tpb, gridSize, stateSize, bitSize, nWrite, height;
	double dx;
	int bks;

	Equation(inputf inFile, str outpath, int argc, char *argv[]);

	~Equation()
	{
		outputf solutionOut(spath.c_str(), std::ofstream::trunc);
		solution["meta"] = inJ;
		solutionOut << solution;
		solutionOut.close();
		delete[] chosenEquation;	
	};


	void makeInitialCondition(states *nState);

	void solutionOutput(states *outState, double tstamp);
};

#endif