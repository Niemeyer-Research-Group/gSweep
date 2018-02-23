#ifndef ROOT_H
#define ROOT_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <iostream>
#include <fstream>
#include <ostream>
#include <istream>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

// UTILITIES PATH
#include "myVectorTypes.h"
#include "json/json.h"
#include "cudaUtils.h"

#define REAL        double
#define ZERO        0.0
#define QUARTER     0.25
#define HALF        0.5
#define ONE         1.0
#define TWO         2.0

typedef Json::Value jsons;
typedef std::istream inputf;
typedef std::ostream outputf;
typedef std::string str;

#endif