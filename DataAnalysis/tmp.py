import sys,os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset

sys.path.append("/home/natacha/Documents/Work/python/")  # to import galib
sys.path.insert(1, os.path.join(sys.path[0], '..'))  #to use DataAnalysis submodules

from galib.tools.nc import print_nc_dict

nc = Dataset("/home/natacha/Documents/Work/Data/Bergen/ec.ens.2020011400.sfc.meteogram.nc","r")

print_nc_dict(nc)

long_0 = nc.variables["longitude"][0]
long_1 = nc.variables["longitude"][-1]
lat_0 = nc.variables["latitude"][0]
lat_1 = nc.variables["latitude"][-1]



