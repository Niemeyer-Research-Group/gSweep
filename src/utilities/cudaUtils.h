#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <vector.h>

void cudaRunCheck()
{
    int rv, dv;
    cudaDriverGetVersion(&dv);
    cudaRuntimeGetVersion(&rv);
    printf("CUDA Driver Version / Runtime Version  --- %d.%d / %d.%d\n", dv/1000, (dv%100)/10, rv/1000, (rv%100)/10);
}

struct cudaTime
{
    std::vector<double> times;
    cudaEvent_t start, stop;
	double ti;
    std::string typ = "GPU";

    cudaTime() {
        cudaEventCreate( &start );
	    cudaEventCreate( &stop );
    }
    ~cudaTime()
    {
        cudaEventDestroy( start );
	    cudaEventDestroy( stop );
    }

    void tinit(){ cudaEventRecord( start, 0); };

    void tfinal() { 
        cudaEventRecord(stop, 0);
	    cudaEventSynchronize(stop);
	    cudaEventElapsedTime( &ti, start, stop);
        ti *= 1.0e3;
        times.push_back(ti); 
    };

    void getLastTime()
    {
        return ti;
    }

    int avgt() { 
        return std::accumulate(times.begin(), times.end(), 0)/ times.size();
        }
};