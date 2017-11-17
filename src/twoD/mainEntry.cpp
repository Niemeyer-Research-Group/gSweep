/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

//COMPILE LINE!
// nvcc -o ./bin/KSOut KS1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -restrict -Xcompiler -fopenmp --ptxas-options=-v

#include "mainGlobals.h"
#include "wave1D.h"

#define GPUNUM          0

// SOME OPTION TO JUST OUTPUT l2 norm of difference
int main(int argc, char *argv[])
{
    std::string ext = ".json";
    std::string sout = argv[3];
    sout.append(ext); 
    std::string scheme = argv[1];

    readIn(argc, argv);
    initConsts();

    cudaSetDevice(GPUNUM);
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    states *state;

    cudaHostAlloc((void **) &state, nX*cGlob.szState, cudaHostAllocDefault);    
    
    for (int k=0; k<2; k++)
    {
        for (int n=0; n<nX; n++) 
        {
            exactSolution(state, n, k); // Macro perhaps
        }
    }

	// This puts the constant part of the equation in constant memory
    cudaMemcpyToSymbol(deqConsts, &heqConsts, sizeof(eqConsts));
    int tstep = TSTEPI;

    cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
    cudaEventRecord( start, 0);
    
	// Call the kernels until you reach the iteration limit.
	double tfm;
	if (!scheme.compare("C"))
    {
		tfm = sweptWrapper(state, &tstep);
    }
    else if (!scheme.compare("C"))
    {
        tfm = classicWrapper(state, &tstep);
    }
    else
    {
        std::cerr << "Incorrect or no scheme given" << std::endl;
    }

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
    cudaEventElapsedTime( &timed, start, stop);
    
    timed *= 1.e3;

    int n_timesteps = tfm/dt;

    double per_ts = timed/(double)n_timesteps;

    cout << n_timesteps << " timesteps" << endl;
    cout << "Averaged " << per_ts << " microseconds (us) per timestep" << endl;
    std::string tpath = pth + "/t" + fspec + ext;

    try {
        std::ifstream tjson(tpath, std::ifstream::in);
        tjson >> timing;
        tjson.close();
    }
    catch (...) {}

    std::string tpbs = std::to_string(cGlob.tpb);
    std::string nXs = std::to_string(cGlob.nX);
    std::cout << cGlob.gpuA << std::endl;

    std::ofstream timejson(tpath.c_str(), std::ofstream::trunc);
    timing[tpbs][nXs] = per_ts;
    timejson << timing;
    timejson.close();

    // end timing out

    std::string spath = pth + "/s" + fspec + "_" + std::to_string(ranks[1]) + ext;
    std::ofstream soljson(spath.c_str(), std::ofstream::trunc);
    solution["meta"] = inJ;
    soljson << solution;
    soljson.close();

    cudaFreeHost(state);
    return 0;
}
