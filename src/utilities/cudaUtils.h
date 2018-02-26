#ifdef GSWEEPUTIL_H
#define  GSWEEPUTIL_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <new>
#include <assert.h>
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

// So set up class to generic equation class, pass in and choose new class.
void setpt(equation *cp)
{
    turn_to<myclass> (cp);
    
}

template <typename TO_T, typename FROM_T>
inline void turn_to(FROM_T* p) 
{ 
    assert( sizeof(FROM_T) == sizeof(TO_T));
    ::new(p) TO_T(); // use of placement new
}

#endif