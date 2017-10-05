
jsons inJ;
jsons solution;
jsons timing;

void parseArgs(int argc, char *argv[])
{
    if (argc>4)
    {
        for (int k=4; k<argc; k+=2)
        {
            inJ[argv[k]] = atof(argv[k+1]);   
        }
    }
}

// Equation, grid, affinity data
void readIn(int argc, char *argv[])
{ 
	ifstream injson(argv[2], ifstream::in);
	injson >> inJ;
	injson.close();
	parseArgs(int argc, char *argv[])
}