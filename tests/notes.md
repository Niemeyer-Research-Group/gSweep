# GTC

- [Application](http://proposals.gputechconf.com/gtc-2018-talks/edit.php?s=Sr4VEPsGPsJxjlrLbPdk)

## Title

- Cooperative Groups and Domain Decomposition for Explicit PDE Solvers

## Talk Description (800 Char)
Discover the swept rule, a new technique for communication avoiding domain decomposition for explicit PDE solvers. This session will build on previous work exploring the effects of memory hierarchy and launch configuration on implicitly synchronized, 1-D solver performance using swept and naïve decomposition schemes by (1) Extending the swept scheme to two dimensions, (2) Comparing the performance of implicit synchronization to in-kernel grid synchronization enabled by the cooperative group interface introduced in CUDA 9. Examples will be provided using the wave equation on the readily extensible, open-source software developed for this project.

## Extended Abstract (3000 Char)

The swept rule exploits the domain of dependence of explicit solver stencils on the space-time grid. It partitions the grid and puts the solution in private memory; as the solution evolves, only the grid points with locally available stencil values move forward in time. When the partition (node) cannot proceed any further it passes the necessary values to a neighboring node and, similarly, receives them.  This greatly reduces inter-node communication events which occur every timestep in a naive decomposition. In a GPU only application, these communication events require global synchronization and involve memory transfer between global and shared or register memory. 

The cooperative group model introduced in CUDA 9 offers the ability to globally sync threads within a kernel which eliminates the need to implicitly synchronize threads by returning control to the host. This could have a large impact on performance, particularly for naive decomposition where the kernel must be relaunched every timestep. 

Our initial 1-D study of the GPU based swept rule using implicit synchronization shows a 6-9x speedup for swept compared to naive decomposition on finite-difference methods applied to scalar equations with small, <10^4 point, grids. This speedup diminishes to 2-3x on large grids, ~10^6 points. When applied to systems of equations requiring more complex discontinuous methods, naive decomposition outperformed the swept rule. 

In this study we will explore the effect of global synchronization on swept and naive decompositions on one and two dimensional stencils. In general, we expect global to outperform implicit synchronization, and that this effect will be more pronounced in naive schemes. We also expect swept rule speedups to diminish in 2-D where greater numbers of spatial points are required. 


Our solver designs balance economical memory usage with modularity and simplicity.  

The frequency of these communication events renders their fixed cost, latency, a significant barrier to performance.  The swept rule is a domain decomposition scheme that arranges computation to avoid communication events and diminish the overall latency cost of a simulation.  The swept rule continues the computation on an individual node whose stencil contains locally available values.  This way the simulation may continue until no spatial points have available stencils and the required values are passed to the neighboring node in a single communication event.  In this study, we apply this scheme to the Euler equations for incompressible flow in one dimension using Sod’s shock tube problem on a cluster at Oregon State University with one GPGPU.  We study the effect of the ratio of CPU to GPU work and the overall performance of the application compared to a naïve scheme.

## Biography (1000 Char | 100 Words)
Daniel is a MS student and graduate research assistant in Mechanical Engineering at Oregon State University under Dr. Kyle Niemeyer conducting research on applying domain decomposition methods for explicit CFD solvers to GPU architecures on their own and in clusters. Daniel holds a B.S. in Mechanical Engineering from Oregon State University and a B.A. in English from Temple University.