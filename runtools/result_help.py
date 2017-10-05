"""
    Classes and functions for plotting the actual results of the simulations.

"""

import numpy as np
import os
import os.path as op
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import json as j
import pandas as pd
import palettable.colorbrewer as pal
import subprocess as sp
import collections
# import datetime.datetime as dt

plt.rc("axes", prop_cycle=cycler("color", pal.qualitative.Dark2_8.mpl_colors))

mpl.rcParams["lines.linewidth"] = 3
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True

def numerica(df):
    df.columns = pd.to_numeric(df.columns.values)
    df.index = pd.to_numeric(df.index.values)
    df.sort_index(inplace=True)
    return df.sort_index(axis=1)

def dictframes(d, t):
    print(t)
    if t>3:
        return {dk: dictframes(d[dk], t-1) for dk in d.keys()}
    else:
        return numerica(pd.DataFrame(d))

def depth(d, level=1):
    if not isinstance(d, dict) or not d:
        return level
    return max(depth(d[k], level + 1) for k in d)

def readj(f):
    fobj = open(f, 'r')
    fr = fobj.read()
    fobj.close()
    return j.loads(fr)

class Solved(object):
   
    def __init__(self, vFile):
        self.ext = ".pdf"
        if isinstance(vFile, str):
            self.jdict = readj(vFile)
        else:
            self.jdict = vFile

        self.deep = depth(self.jdict)
        self.ddf = dictframes(self.jdict, self.deep)
        self.subpl = len(self.jdict.keys())

    def metaparse(self, probspec):
        self.pr = list(probspec.keys())[0]
        pdi = probspec[self.pr]
        self.plotname = self.pr + "_" + str(pdi["nX"])
        
        # self.vals = np.genfromtxt(dataTuple, skip_header=1)[:,2:]
        # self.varNames = np.genfromtxt(dataTuple, skip_header=1, dtype="string")[:,0]
        # self.tFinal = np.around(np.genfromtxt(dataTuple, skip_header=1)[:,1], decimals=7)
        # self.utFinal = np.unique(self.tFinal)
        # self.plotTitles = np.unique(self.varNames)
        # self.plotname = self.datafilename.split("_")[0]
        # self.subpl = "Euler" in self.plotname            

    def stripInitial(self):
        stripped = collections.defaultdict(dict)
        for i, t in enumerate(self.tFinal):
            if t == 0:
                continue
            
            stripped[self.varNames[i]][t] = self.vals[i,:]

        return stripped
        
    def plotResult(self, f, a):   
        
        if (type(a).__module__ != np.__name__):
            a = np.array(a)
            
        a=a.ravel()
            
        for k, axx in zip(self.ddf.keys(), a):
            dfg = self.ddf[k]
            dfg.plot(ax=axx)
            if self.subpl == 1:
                axx.set_ylabel(k)
            else:
                axx.set_title(k)
                
#        hand, lbl = a[0].get_legend_handles_labels()
#        f.legend(hand, lbl, loc="upper right", fontsize="medium")
        if self.subpl == 1:
            f.subplots_adjust(bottom=0.08, right=0.85, top=0.9, 
                    wspace=0.15, hspace=0.25)

#        else:          
#            for axi, nm in zip(ax, self.plotTitles):
#                idx = np.where(self.varNames == nm)
#                vn = self.vals[idx, :].T
#                tn = self.tFinal[idx]
#                for i, tL in enumerate(tn):
#                    axi.plot(self.xGrid, vn[:,i], label="{:.3f} (s)".format(tL))


    def annotatePlot(self, fh, ax):

        if not self.subpl:

            ax.set_ylabel(self.plotTitles[0])
            ax.set_xlabel("Spatial point")
            ax.set_title(self.plotname + " {0} spatial points".format(self.numpts))
            hand, lbl = ax.get_legend_handles_labels()
            fh.legend(hand, lbl, loc="upper right", fontsize="medium")
        
        else:
            fh.suptitle(self.plotname + 
                " | {0} spatial points   ".format(self.numpts), 
                fontsize="large", fontweight="bold")

            for axi, nm in zip(ax, self.plotTitles):
                axi.set_title(nm)
                
            hand, lbl = ax[0].get_legend_handles_labels()
            fh.legend(hand, lbl, loc="upper right", fontsize="medium")

            fh.subplots_adjust(bottom=0.08, right=0.85, top=0.9, 
                                wspace=0.15, hspace=0.25)

    def savePlot(self, fh, plotpath):
        
        plotfile = op.join(plotpath, self.plotname + self.ext)
        fh.savefig(plotfile, dpi=200, bbox_inches="tight")

    def gifify(self, plotpath, fh, ax):

        self.ext = ".png"
        ppath = op.join(plotpath, "Temp")
        os.chdir(ppath)
        giffile = op.join(plotpath, self.plotname + ".gif")
        avifile = op.join(ppath, self.plotname + ".avi")
        pfn = "V_"
        ptitle = self.plotname
        mx = np.max(self.vals[1,:])
        mn = np.min(self.vals[1,:])
        if not self.subpl:
            for i, t in enumerate(self.tFinal):       
                ax.plot(self.xGrid, self.vals[i,:])
                self.plotname = pfn + str(i)
                ax.set_ylabel(self.plotTitles[0])
                ax.set_xlabel("Spatial point")
                ax.set_title(ptitle + " {0} spatial points : t = {1} (s)".format(self.numpts, t))
                ax.set_ylim([mn, mx+2])
                self.savePlot(fh, ppath)
                
                ax.clear()
                
        else:
            for i, t in enumerate(self.utFinal):
                idx = np.where(self.tFinal == t)
                v = self.vals[idx, :]
                nom = self.varNames[idx]       
                for axi, nm in zip(ax, self.plotTitles):
                    idx = np.where(nom == nm)
                    vn = v[idx, :].T
                    
                    axi.plot(self.xGrid, vn)

                self.plotname = pfn + str(i)
                ax.set_ylabel(self.plotTitles[0])
                ax.set_xlabel("Spatial point")
                ax.set_title(self.plotname + " {0} spatial points : t = {1}".format(self.numpts, t))
                self.savePlot(fh, ppath)

                for a in ax:
                    a.clear()

        # st = "linux"
        # if st in sys.platform:
        #     try (sp.call(["ffmpeg", "-i", "V_%d.png", "-r", "4", avifile])):
        #         print("Writing avi")
        #         sp(call(["ffmpeg", "-i", avifile, giffile])
        #         print("Writing gif")
        #     except:
        #         print("------------------")
        #         print( "Install ffmpeg with: sudo apt-get install ffmpeg")
        #         f = os.listdir(".")
        #         for fm in f:
        #             os.remove(fm)
                    
        #         raise SystemExit

        #     f = o(.listdir(".")
        #     for fm in f:
        #         o.remove(fm)
        # else
        #     print("------------------")
        #     print( "This script only makes gifs on linux with ffmpeg.")
        #     print("The images are still in the folder under ResultPlots/Gifs/Temp.")
            
        

        


        
        
