/** 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

//COMPILE LINE!
// nvcc -o ./bin/KSOut KS1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -restrict -Xcompiler -fopenmp --ptxas-options=-v

#include "wave1D.h"

// SOME OPTION TO JUST OUTPUT l2 norm of difference

int main(int argc, char *argv[])
{
    // Command line input variables.
    cGlob.dt = argv[1]; 

    cudaSetDevice(GPUNUM);
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    states *state;

    cudaHostAlloc((void **) &state, nX*sizeof(states), cudaHostAllocDefault);    
    
    for (int k=0; k<2; k++)
    {
        for (int n=0; n<nX; n++) 
        {
            exactSolution(state, n, k); // Macro perhaps
        }
    }

	// This puts the constant part of the equation in constant memory
	cudaMemcpyToSymbol(deqConsts, &heqConsts, sizeof(eqConsts));

    cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
    cudaEventRecord( start, 0);
    
	// Call the kernels until you reach the iteration limit.
	double tfm;
	if (scheme)
    {
		tfm = sweptWrapper();
	}
	else
	{
		tfm = classicWrapper();
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

    if (argc>7)
    {
        ofstream ftime;
        ftime.open(argv[8],ios::app);
        ftime << dv << "\t" << tpb << "\t" << per_ts << endl;
        ftime.close();
    }

    return 0;
}
