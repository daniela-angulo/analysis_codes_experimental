import numpy as np
import matplotlib as mpl 
from scipy.optimize import curve_fit
from matplotlib import pylab as plt
from numpy import linalg
import csv
from nptdms import TdmsFile
from scipy import special
from matplotlib import rc
plt.ioff()

mpl.rcParams.update({'font.size': 14,'text.usetex':True})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

det=['mp3','mp2','mp15','mp1','mp07','mp05','mp04','mp03','p04','p05','p07','p1','p15','p2','p3']

dir="analysis_20210903/"
num_steps=10

XPS_matrix=np.zeros([len(det),num_steps])

for i in range(len(det)):
	XPS_matrix[i]=np.loadtxt(dir+det[i]+'XPS_array.csv',delimiter = ',')

np.savetxt(dir+'XPS_matrix',(XPS_matrix))