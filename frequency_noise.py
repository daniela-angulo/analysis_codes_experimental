import matplotlib.pyplot as plt
import matplotlib as mpl
from mayavi import mlab
import numpy as np
import time as time
mpl.rcParams.update({'font.size': 15,'font.family': 'STIXGeneral','mathtext.fontset': 'stix'})
mpl.interactive(True)

def P_detuning(x,center,width):
	return (1./np.sqrt(2*np.pi*width**2))*np.exp(-(x-center)**2/(2.*width**2))

def P_click_detuning(x,OD_0,gamma):
	OD=OD_0/(1+(2*x/gamma)**2)
	return np.exp(-OD)

def Linear_phase(x,OD_0,gamma):
	return (2*OD_0/gamma)*x/(1+(2*x/gamma)**2)

def cvnc2(probe_det,shift_array,OD_0,gamma):
	return (Linear_phase(probe_det+shift_array,OD_0,gamma)-Linear_phase(probe_det,OD_0,gamma))

#frequency in MHz
width=.3
gamma=6.
OD_0=1.
points=100
start=-7.
stop=-start
x=np.linspace(start,stop,points)
probe_det=3.5
shift_array=[]
center_array=[]
arr=[]
#integral 
for i in range(100):
	center=-5+i*0.1
	z=x*P_detuning(x,center,width)*P_click_detuning(x,OD_0,gamma)
	normalization=((stop-start)/points)*np.sum(P_detuning(x,center,width)*P_click_detuning(x,OD_0,gamma))
	mean=((stop-start)/points/normalization)*np.sum(z)
	signal_shift=mean-center
	center_array=np.append(center_array,center)
	arr=np.append(arr,100*(Linear_phase(x+signal_shift,OD_0,gamma)-Linear_phase(x,OD_0,gamma)))
	shift_array=np.append(shift_array,signal_shift)


arr=np.reshape(arr,(100,100))
cvnc=Linear_phase(x+signal_shift,OD_0,gamma)-Linear_phase(x,OD_0,gamma)
# fig1,axes1=plt.subplots(2,2,figsize=(12,8))
# axes1[0,0].plot(center_array,100*cvnc2(probe_det,shift_array,OD_0,gamma),'c' ,color = 'c')
# axes1[0,0].plot(np.append(-np.flip(center_array),center_array),100*cvnc2(-probe_det,shift_array,OD_0,gamma),'c' ,color = 'r')
# axes1[0,0].plot(np.append(-np.flip(center_array),center_array),100*cvnc2(0,shift_array,OD_0,gamma),'c' ,color = 'g')
# axes1[0,0].set_xlabel(r'$\Delta_{s}$')
# axes1[0,0].set_ylabel(r'CvNC phase [mrad]')
# axes1[0,0].grid() 
# axes1[0,1].plot(x,P_click_detuning(x,OD_0,gamma), 'c', color = 'c')
# axes1[0,1].set_xlabel(r'$\Delta_{s}$')
# axes1[0,1].set_ylabel(r'$P(click|\Delta_{s})$')
# axes1[0,1].grid()
# axes1[1,0].plot(x,P_detuning(x,center,width)*P_click_detuning(x,OD_0,gamma)/normalization, 'c', color = 'c')
# axes1[1,0].set_xlabel(r'$\Delta_{s}$')
# axes1[1,0].set_ylabel(r'$P(\Delta_{s}|click)$')
# axes1[1,0].axvline(x=mean,color='orange',linewidth=2.0)
# axes1[1,0].grid()
# axes1[1,1].plot(x,100*cvnc, 'c', color = 'c')
# axes1[1,1].set_xlabel(r'$\Delta_{p}$')
# axes1[1,1].set_ylabel(r'CvNC phase [mrad]')
# axes1[1,1].grid() 

plot = mlab.surf(center_array,x,arr, warp_scale = 'auto')
mlab.axes(plot, color =(0.7,.7,.7), xlabel = 'signal detuning', ylabel = 'probe detuning', zlabel = 'F')
mlab.show()