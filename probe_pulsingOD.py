import numpy as np
import matplotlib as mpl 
from matplotlib import pylab as plt
import time as time
from scipy.optimize import curve_fit
from numpy import linalg
import csv
import os
import sys
from nptdms import TdmsFile
from scipy import special

starttime = time.time()
mpl.rcParams.update({'font.size': 12, 'font.family': 'STIXGeneral',
                            'mathtext.fontset': 'stix'})

def inverse(x,m,b):
	return m/x+b

def exponential(x,m,a,b):
	return np.exp(-m*(x-a))+b
main_dir="D:/Data/20211025/XPS_pdet/mp15"

stringname1=main_dir+"amp_pulsing"
final_amp=np.genfromtxt(stringname1+".csv",delimiter = ',')
x_arr=np.arange(len(final_amp))

stringname=main_dir+"_refamp_pulsing"
reference=np.genfromtxt(stringname+".csv",delimiter = ',')

probePulsing_OD=-2*np.log(final_amp/reference)
#popt1,pcov1= curve_fit(exponential,x_arr,y)
#bounds=([-2.e-2,62800,-7], [2.e-2,62850,7])

print(np.mean(probePulsing_OD[200:3000]))
plt.figure()
plt.plot(x_arr,probePulsing_OD,'ro')
#plt.plot(x_arr,y,'ro',x_arr,exponential(x_arr,*popt1),'-')
#plt.plot(x,1000*inverse(np.sqrt(photon_number),0.5,0),'go')
#plt.plot(x_coordinates,y_coordinates)
plt.xlabel(r'measurement')
plt.ylabel(r'probe OD')
plt.ylim(bottom=0)
plt.xlim(left=0)
#plt.legend((r'Data',r'exp(-%.2e(x-%.2e))+%5.2f'%tuple(popt1)))
plt.grid()
plt.savefig(stringname1+'OD.png',dpi=400)

np.savetxt(stringname1+'OD.csv',probePulsing_OD,delimiter = ',')

#plot ratio of XPS to OD
stringname1=main_dir+"XPS_shots_average"
XPS=np.genfromtxt(stringname1+".csv",delimiter = ',')
average_OD=np.mean(np.reshape(probePulsing_OD,(int(len(XPS[0])),int(len(probePulsing_OD)/len(XPS[0])))),1)
ratio=(XPS[0]/average_OD)
plt.figure()
plt.plot(np.arange(len(average_OD)),ratio,'ro')
plt.title(r'ratio XPS to OD'+stringname1)
plt.xlabel(r'measurement')
plt.ylabel(r'ratio XPS/OD')
plt.xlim(left=0)
plt.grid()
plt.savefig(stringname1+'ratio.png',dpi=400)
plt.show()