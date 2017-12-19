import os
import os.path as op
import sys
import numpy as np

subGrid = (16, 16)
grid = (4, 4)
wholeGrid = subgrid[0]*grid[0] + subgrid[1]*grid[1] 

def getBlock(gr):
    pass

#r in the x direction 0.25-1
#b in the y direction 0.25-1

xWorking, yWorking = np.meshgrid(np.arange(wholeGrid[0]),np.arange(wholeGrid[0]))
xTid = xWorking/grid[0]
yTid = yWorking/grid[1]

# it = np.nditer(workGrid, flags=['multi_index'])
# while not it.finished:
#     tidGrid[it.multi_index] = (it.multi_index[0]/grid[0], it.),
#     it.iternext()

