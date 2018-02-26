/** 
    Copyright (C) 2018 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

//COMPILE LINE!

#include "kernel.h"
#include "coopKernel.h"
#include "cudaUtils.h"

#define GPUNUM          0

// SOME OPTION TO JUST OUTPUT l2 norm of difference

// Path
int main(int argc, char *argv[])
{
    // No dlopen :(, if dlopen this is 2
    cudaRunCheck();
    int clargStart = 1;
    
    str cooperation = argv[clargStart + 0];
    str scheme = argv[clargStart + 1];
    str infile = argv[clargStart + 2];
    str outpath = argv[clargStart + 2];

    Equation theEquation(infile, outpath, argc-3, &argv[clargStart+3]);

    Solver solveit(&theEquation, scheme, cooperation);

    solveit.solveEquation();

    solveit.writeFiles();

    return 0;
}
