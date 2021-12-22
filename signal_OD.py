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

stringname="D:/Data/20211021/signal_OD/noprobeCWclicks_cycle"
final_amp=np.genfromtxt(stringname+".csv",delimiter = ',')
x_arr=np.arange(len(final_amp))

stringname="D:/Data/20211021/signal_OD/noprobeCW_refclicks_cycle"
reference=np.genfromtxt(stringname+".csv",delimiter = ',')

signalPulsing_OD=-np.log(final_amp/np.mean(reference))
#popt1,pcov1= curve_fit(exponential,x_arr,y)
#bounds=([-2.e-2,62800,-7], [2.e-2,62850,7])

#print(np.mean(probePulsing_OD[200:3000]))

plt.plot(x_arr,signalPulsing_OD,'ro')
#plt.plot(x_arr,y,'ro',x_arr,exponential(x_arr,*popt1),'-')
#plt.plot(x,1000*inverse(np.sqrt(photon_number),0.5,0),'go')
#plt.plot(x_coordinates,y_coordinates)
plt.xlabel(r'measurement')
plt.ylabel(r'signal OD')
plt.ylim(bottom=0)
plt.xlim(left=0)
#plt.legend((r'Data',r'exp(-%.2e(x-%.2e))+%5.2f'%tuple(popt1)))
plt.grid()
plt.savefig(stringname+'OD.png',dpi=400)
plt.show()