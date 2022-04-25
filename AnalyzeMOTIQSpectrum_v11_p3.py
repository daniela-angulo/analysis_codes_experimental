"""
----------------------------------------------------------------
Created: Mar 17, 2021
By: Kyle Thompson

Description: This is an analysis code for the Spectrum DAQ card. It is based on a copy of AnalyzeMOTIQSpectrum_v0.
This version can handle digital data (stored in the ch1 analog data). Also getting rid of all g2 stuff to make
this code a bit neater, and getting rid of anything having to do with the second digital data channel (used for
the trigger on the Picoscope).

Updated: Oct, 2021 (v7)
By: Daniela

Description: What Kyle said. Now we are going to clean up the code. We want to use more points for signal pulses by getting rid of the EIT scan (3000 more). 
This card allows us to get longer cycles but this code is only for the 72000 measurements. 
I deleted the autocorrelation code from line 427 because we never use it and in case we want that part again, we can bring it back from the old versions. 
I got rid of the part HISTOGRAM OF PHASE SHIFT IN A SHOT GIVEN A CLICK and phase shift.
The variable std_dir_test[k]=phase_std2 is taking the standard deviation of 100 phase points. This was created to study the phase noise when varying the power. I'll comment it for now.
I'm going to change the way we calculate the means by just doing big sums and keeping track of how many shots total.
The spectrum feature really slows down the code and we need it. Let's fix it by putting the conditional outside the for loop. 
v6 Let's add the shots from the EIT window. 6000 points or how ever many can complete shots of 36. 
v7 I added a way to see the XPS over an atom cycle after averaging all the atom cycles. Line 323 saving amplitudes to find the OD during the pulsing time.
These recent versions have been about analyzing different quantities over an atom cycle. Averaging all the atom cycles first and then breaking it in chunks. See last 20 lines. 
v10 to only look at the second half of the cycle or just divide it into two 
v10p1 long shots:  added capability to run program while simultaneously saving data from the VI (Vida)
v11 sorry about everything I'm gonna get rid of here but this code is so long and so hard to look at that it needs to be shrunk. I took away a bunch of plots that were very useless and now I'm gonna modify the first one so it shows the average and not the amplitude of a single file
----------------------------------------------------------------
"""

"""
This comes from Josiah's and then Kyle's code. 

"""
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

from Spectrum_info_class_p3 import Spectrum_info_class

starttime = time.time()
mpl.rcParams.update({'font.size': 10, 'font.weight': 'bold','font.family': 'STIXGeneral',
                            'mathtext.fontset': 'stix'})


dir_main = 'D:/Data/20220420/CvNC_bypassingatoms/40ns_FG_modulation_8_1917'
dir_main = 'sample_data_clicks'

fileendstring = '_0.tdms'

analyze_while_taking_data = True
number_of_files = 1000 # Number of files that the program expects you to eventually save when "analyze_when_taking_data" is enabled

AnalyzeDigitalDataBool = True
TemporalFilteringBool = False
ReplaceBool = False
Spectrum = True
correction_factor = 6.605 #see evernote from Mar 1, 2021 for calibration
delayshift = 0#delay the SPCM's by some # of shots to compensate for a 1us delay with BNCs
#factor=1 #1.70*.94, this has to be checked every day (don't know)
zerovalue = 2.75 #this has to be checked every day for the amplitude in mV (don't know)
#BackgroundPhase=0 #np.genfromtxt('backgroundphaseATOMS20190823.txt', delimiter=',') (don't know)
numShots = 1648#2*2*375
numMeasurementsProbe1 = int(59328)
numMeasurementsTotal =  int(72000)
numMeasurementsPerShot = int(144/4) #thats 2382.25ns
scansize = int(6300)

#these have to be equidistant etc otherwise the fit approach won't match the subtraction approach

# for 50ns Gaussian pulses
shift = 2 #1
start1 = 3+shift
stop1 = 9+shift
start2 = 17+shift #9
stop2 = 18+shift #10
start3 = 26+shift #17
stop3 = 32+shift #21

# # for 10ns Gaussian pulses
# shift = -4 #2
# start1 = 6+shift
# stop1 = 10+shift
# start2 = 17+shift #9
# stop2 = 18+shift #10
# start3 = 25+shift #17
# stop3 = 29+shift #21


def sixteenBitIntegerTomV(Array,VoltageRange):
	Multiplier = VoltageRange/32767.0
	Array_involts = Multiplier*Array
	return(Array_involts)

def separate_analog_and_digital(combined_data):
	#splits data from ch1 into the analog value and the digital part
	# digital_data = np.bitwise_and(1, np.right_shift(combined_data,15)) #pretty fast, but there may be some faster option
	# analog_data = np.left_shift(np.bitwise_and(combined_data,0x7fff), 1)

	digital_data = (combined_data >> 15) & 1 #pretty fast, but there may be some faster option
	analog_data = (combined_data & 0x7fff) << 1
	return(analog_data,digital_data) 

#load data function
def Load_Data(file1):
	tdms_file = TdmsFile(file1)
	file_info = Spectrum_info_class(file1) #get info about the file (measurement settings, etc)
	num_channels = file_info.num_channels_enabled
	interleaved = tdms_file['Raw Data']['Channel 0']#c0s0,c1s0,c0s1,c1s1,c0s2,c1s2, etc (c=channel, s=sample)
	interleaved_data = interleaved.data #i16 integer
	data0, data1 = np.reshape(interleaved_data, (num_channels,int(len(interleaved_data)/num_channels)), 'F') #the F here is needed to deal with the interleaved format of the data

	#convert SPCM data into what channel fired when...
	if AnalyzeDigitalDataBool == True:
		data1, digital_data = separate_analog_and_digital(data1) #extract digital bits from analog data (they're stored in there), and replace data1 with analog data
	else:
		digital_data = np.zeros(len(data1)) #set digital to 0's and leave data1 alone

	ch1_x = np.arange(len(data1))
	I = data0.astype(int)
	Q = data1.astype(int)
	
	#IQ conversion
	amplitude = correction_factor*np.sqrt(I**2+Q**2) #correction_factor is the conversion #got rid of the factor of 2 here (Kyle, Feb 24, 2021)

	phase_wrapped2 =  np.arctan(Q/I)
	phase_wrapped2[I == 0] = np.pi/2
	phase = np.unwrap(phase_wrapped2,discont=np.pi/2,period=np.pi)

	#Warnings
	if len(ch1_x) != numMeasurementsTotal*file_info.num_segments:
		print('Warning, trace is the wrong length')
	return(ch1_x, sixteenBitIntegerTomV(amplitude,file_info.ch0_voltage_range_number), phase,digital_data, file_info)


def Analyze(amplitude, phase, digitaldata1, numShots, numMeasurements, numMeasurementsPerShot):
	#take 27000 long array and make it into 375, 72 long arrays
	if len(amplitude) != numMeasurements:
		print('Warning, array is the wrong length')
	Phase_in_a_shot = phase.reshape(numShots,numMeasurementsPerShot)
	Amplitude_in_a_shot = amplitude.reshape(numShots,numMeasurementsPerShot)
	
	#subtract off linear background
	background_i_phase = np.mean(Phase_in_a_shot[:,start1:stop1],1)
	background_f_phase = np.mean(Phase_in_a_shot[:,start3:stop3],1)
	background_phase =  (background_i_phase+background_f_phase)/2
	signal_phase = np.mean(Phase_in_a_shot[:,start2:stop2],1) #getting peak value
	phase_shift = signal_phase - background_phase

	if AnalyzeDigitalDataBool == True:
		DigitalData_in_a_shot = digitaldata1.reshape(numShots,numMeasurementsPerShot)
		#This is done to only count the arrival time bin of a click 
		b=np.zeros_like(DigitalData_in_a_shot)
		b[np.arange(len(DigitalData_in_a_shot)), DigitalData_in_a_shot.argmax(1)] = 1 #python doesn't know what to do when all the elements are equal and returns the first index
		b[:,0]=0 #I do this to set that first index back to zero in the rows that only contain zeros, the other rows don't have a problem
		DigitalData1_in_a_shot=b
		if TemporalFilteringBool == True:
			todelete = np.concatenate((np.arange(0,start2-3,1),np.arange(stop2+3,numMeasurementsPerShot,1)))
			DigitalData1_in_a_shot = np.delete(DigitalData1_in_a_shot,todelete,axis=1) #this is for temporally filtering the SPCMs
		DigitalData1 = np.roll(np.any(DigitalData1_in_a_shot,1),delayshift)
		digitalChannel1Counter=np.sum(DigitalData1)

		if np.sum(DigitalData1)>0:
			Phase_in_a_shot_CLICK = Phase_in_a_shot[DigitalData1]
			Phase_in_a_shot_NOCLICK = Phase_in_a_shot[np.logical_not(DigitalData1)]

			background_i_phase_CLICK = np.mean(Phase_in_a_shot_CLICK[:,start1:stop1],1)
			background_f_phase_CLICK = np.mean(Phase_in_a_shot_CLICK[:,start3:stop3],1)
			background_phase_CLICK =  (background_i_phase_CLICK+background_f_phase_CLICK)/2
			signal_phase_CLICK = np.mean(Phase_in_a_shot_CLICK[:,start2:stop2],1)
			phase_shift_CLICK = signal_phase_CLICK - background_phase_CLICK
	
			background_i_phase_NOCLICK = np.mean(Phase_in_a_shot_NOCLICK[:,start1:stop1],1)
			background_f_phase_NOCLICK = np.mean(Phase_in_a_shot_NOCLICK[:,start3:stop3],1)
			background_phase_NOCLICK =  (background_i_phase_NOCLICK+background_f_phase_NOCLICK)/2
			signal_phase_NOCLICK = np.mean(Phase_in_a_shot_NOCLICK[:,start2:stop2],1)
			phase_shift_NOCLICK = signal_phase_NOCLICK - background_phase_NOCLICK

		else:
			phase_shift_CLICK=phase_shift
			phase_shift_NOCLICK=phase_shift
			Phase_in_a_shot_CLICK=np.zeros((numShots,numMeasurementsPerShot))
			Phase_in_a_shot_NOCLICK=np.zeros((numShots,numMeasurementsPerShot))

	else:
		DigitalData1=np.zeros(numShots)
		DigitalData1_in_a_shot=np.zeros((numShots,numMeasurementsPerShot))
		digitalChannel1Counter=0
		phase_shift_CLICK=phase_shift
		phase_shift_NOCLICK=phase_shift
		Phase_in_a_shot_CLICK=np.zeros((numShots,numMeasurementsPerShot))
		Phase_in_a_shot_NOCLICK = np.zeros((numShots,numMeasurementsPerShot))

	return(Phase_in_a_shot, Amplitude_in_a_shot, Phase_in_a_shot_CLICK,Phase_in_a_shot_NOCLICK,DigitalData1_in_a_shot, phase_shift,phase_shift_CLICK,phase_shift_NOCLICK,digitalChannel1Counter)

def fit_to_a_line(x,y):
	def func(x,m,b):
		return m*x + b
	popt_func,pcov_func = curve_fit(func,x,y) #bounds = ([-.00031,-2],[-0.0001,1])
	return(popt_func[0],popt_func[1])

#preparing some variables which will be filled with info from all the files
DigitalChannel1Counter_dir = 0 #counts the total number of clicks
averaged_amplitude_in_a_shot_for_dir = 0 
averaged_phase_in_a_shot_for_dir = 0
averaged_phase_in_a_shot_CLICK_for_dir = 0
averaged_phase_in_a_shot_NOCLICK_for_dir = 0
phase_shiftProbe1_File=0
square_phase_shift=0
phase_shiftProbe1_1=0
phase_shift_CLICKProbe1_File=0
phase_shift_NOCLICKProbe1_File=0
numAtomCycles = 0
amplitudeAVG=0
phaseAVG = 0
phase_in_a_cycle=0
DigitalData1_cycles=0
phase_in_a_shot_CLICK_sum=0
square_phase_in_a_shot_CLICK_sum=0
phase_in_a_shot_NOCLICK_sum=0
square_phase_in_a_shot_NOCLICK_sum=0

amplitudeMOT = np.zeros(scansize)
phaseMOT = np.zeros(scansize)
phaseref = np.zeros(scansize)
# amplitude_files=np.zeros([100,numMeasurementsProbe1])


print("Analyzing data in directory: " + dir_main)
# new change
if analyze_while_taking_data == True:
	print("Expecting {} imcoming data files to analyze....".format(number_of_files))
	numFiles = number_of_files
	phase_shift_dir = np.zeros(int(numFiles))
	phase_shift_CLICK_dir = np.zeros(int(numFiles))
	phase_shift_NOCLICK_dir = np.zeros(int(numFiles))
	OD_dir = np.zeros(int(numFiles))
	i=0
	# new change
	for fs in range(0,number_of_files): 	# Iterates through each file in the given directory via expected filename
		filepath = dir_main + "/test_{}".format(fs) + ".tdms"

		# Errors for when we inevitably have type in the wrong number of files with this section of the code enabled.
		if os.path.isfile(filepath) == False:
			if fs == 0:
				print("Error: expected files not found. Program failed to run.")
				print("Exiting UNGRACEFULLY")
				exit(0)
			else:
				filepath = dir_main + "/test_{}".format(fs-1)
				print("Error: only found {} out of {} files. Continuing analysis.".format(fs, number_of_files))
				numFiles = fs # fs number of files, because started at zero and failed at fs+1
				phase_shift_dir = phase_shift_dir[:fs]
				phase_shift_CLICK_dir = phase_shift_CLICK_dir[:fs]
				phase_shift_NOCLICK_dir = phase_shift_NOCLICK_dir[:fs]
				OD_dir = OD_dir[:fs]
				#print("Exiting GRACEFULLY")
				break
		else:
			#print(filepath)
			print("test_{}".format(fs), end='\r')	

		#in each iteration of the loop do stuff to a different file. 
		ch1_x, amplitude, phase,digital_data, file_info  =  Load_Data(filepath)
		SegmentsPerFile = file_info.num_segments #get num segments from file info
		#std_dir_test = np.zeros(SegmentsPerFile)
		amp_dir_test=np.zeros([SegmentsPerFile,numMeasurementsProbe1]) #contains all the amplitude data after the scans in a file, this is to extract the OD within the atom cycle
		if ReplaceBool == True:
			probability = .2/numMeasurementsPerShot
			digitaldatalength = len(digital_data)
			digital_data = np.random.binomial(1,probability,len(digital_data))#use this if you want a random set of digital data
		if Spectrum == True:
			amplitudeAVG += np.mean(amplitude.reshape(SegmentsPerFile,numMeasurementsTotal),0)
			phaseAVG += np.mean(phase.reshape(SegmentsPerFile,numMeasurementsTotal),0)[0:scansize*2] 
			phase_in_a_cycle+=np.mean(phase.reshape(SegmentsPerFile,numMeasurementsTotal),0)[(2*scansize+72):numMeasurementsTotal]
		OD_file=0
		for k in range(SegmentsPerFile):
			startref = 0 + numMeasurementsTotal*k
			stopref = scansize+ numMeasurementsTotal*k
			startMOT = scansize+ numMeasurementsTotal*k
			stopMOT = 2*scansize+ numMeasurementsTotal*k
			startPulsingProbe1 = 2*scansize+72+numMeasurementsTotal*k
			stopPulsingProbe1 = 2*scansize+72+numMeasurementsProbe1+ numMeasurementsTotal*k
			phase_slope,phase_offset= fit_to_a_line(ch1_x[startPulsingProbe1:stopPulsingProbe1],phase[startPulsingProbe1:stopPulsingProbe1])
			ch1PulsingProbe1_x = ch1_x[startPulsingProbe1:stopPulsingProbe1]
			amplitudePulsingProbe1 = amplitude[startPulsingProbe1:stopPulsingProbe1]
			phasePulsingProbe1 = phase[startPulsingProbe1:stopPulsingProbe1]-phase_slope*ch1_x[startPulsingProbe1:stopPulsingProbe1]
			#phasePulsingProbe1 = phase[startPulsingProbe1:stopPulsingProbe1]
			phase_std2 = 1000*np.std(phasePulsingProbe1[10000:10100]-(phase_slope*ch1_x[startPulsingProbe1+10000:startPulsingProbe1+10100])) #calculate the phase noise for 100 points subtracting the slope
			#std_dir_test[k]=phase_std2
			digitaldatapulsingProbe1 = digital_data[startPulsingProbe1:stopPulsingProbe1]
			#Feed the data into the Analyze function. 
			Phase_in_a_shotProbe1, Amplitude_in_a_shotProbe1,Phase_in_a_shot_CLICKProbe1,Phase_in_a_shot_NOCLICKProbe1,DigitalData1_in_a_shotProbe1,phase_shiftProbe1,phase_shift_CLICKProbe1,phase_shift_NOCLICKProbe1,digitalChannel1CounterProbe1 = Analyze(amplitudePulsingProbe1,phasePulsingProbe1,digitaldatapulsingProbe1, numShots, numMeasurementsProbe1, numMeasurementsPerShot)
			#what do we want to keep track of across all files?
			mean_amplitude = np.mean(amplitude[0:700])-zerovalue
			mean_amplitude2 = np.mean(amplitudePulsingProbe1[300:2000])-zerovalue
			OD_file += -2*np.log(mean_amplitude2/mean_amplitude)
			#this is shot information
			averaged_amplitude_in_a_shot_for_dir += np.mean(Amplitude_in_a_shotProbe1,0)
			averaged_phase_in_a_shot_for_dir += np.mean(Phase_in_a_shotProbe1,0)
			averaged_phase_in_a_shot_CLICK_for_dir+=np.mean(Phase_in_a_shot_CLICKProbe1,0)
			averaged_phase_in_a_shot_NOCLICK_for_dir+=np.mean(Phase_in_a_shot_NOCLICKProbe1,0)
			DigitalChannel1Counter_dir += digitalChannel1CounterProbe1 #counting clicks
			DigitalData1_cycles+=DigitalData1_in_a_shotProbe1
			amp_dir_test[k]=amplitudePulsingProbe1
			numAtomCycles += 1
			
			#Find the STD of each point of CvNC using sums 
			phase_in_a_shot_CLICK_sum+=np.sum(Phase_in_a_shot_CLICKProbe1-(np.mean(Phase_in_a_shot_CLICKProbe1)),0)
			square_phase_in_a_shot_CLICK_sum+=np.sum((Phase_in_a_shot_CLICKProbe1-(np.mean(Phase_in_a_shot_CLICKProbe1)))**2,0)
			phase_in_a_shot_NOCLICK_sum+=np.sum(Phase_in_a_shot_NOCLICKProbe1-(np.mean(Phase_in_a_shot_NOCLICKProbe1)),0)
			square_phase_in_a_shot_NOCLICK_sum+=np.sum((Phase_in_a_shot_NOCLICKProbe1-(np.mean(Phase_in_a_shot_NOCLICKProbe1)))**2,0)

			#Here we are trying to collect all the XPS statistics using sums to find std and mean
			phase_shiftProbe1_1+=np.sum(phase_shiftProbe1)/numShots
			square_phase_shift+=np.sum(phase_shiftProbe1**2)/numShots
			phase_shiftProbe1_File+=np.mean(phase_shiftProbe1)/SegmentsPerFile
			phase_shift_CLICKProbe1_File+=np.mean(phase_shift_CLICKProbe1)/SegmentsPerFile #thats the average phase shift in an atom cycle for every atom cycle
			phase_shift_NOCLICKProbe1_File+=np.mean(phase_shift_NOCLICKProbe1)/SegmentsPerFile

			#now pick a random file to make some plots of so we can sanity check everything. 
			if filepath.endswith(fileendstring) and k == 0:		# new change
			#if 1000*np.max(np.mean(phase_shift))>10:
				print("This file analyzed")
				phase_std = 1000*np.std(phase[20000:21000])
				phase_mean = np.mean(phase[12000:13000])
				phase_shift_std = 1000*np.std(phase_shiftProbe1)
				meanphaseshiftforrandomfile = 1000*np.mean(phase_shiftProbe1)
				fig1,axes1 = plt.subplots(3,2,figsize=(12,12))
				#plot the phase across one file
				# ####################################PLOT OF PHASE FROM RANDOM ATOM CYCLE############### -phase_mean
				# axes1[0,1].plot(ch1_x[0:numMeasurementsTotal],phase[0:numMeasurementsTotal]-(phase_slope*ch1_x[0:numMeasurementsTotal]),'-',color='navy',label="Phase",linewidth=3.0)			
				# axes1[0,1].axvline(x=startref,color='orange',linewidth=2.0)
				# axes1[0,1].axvline(x=stopref,color='orange',linewidth=2.0)
				# axes1[0,1].axvline(x=startMOT,color='orange',linewidth=2.0)
				# axes1[0,1].axvline(x=stopMOT,color='orange',linewidth=2.0)				
				# axes1[0,1].axvline(x=startPulsingProbe1,color='orange',linewidth=2.0)
				# axes1[0,1].axvline(x=stopPulsingProbe1,color='orange',linewidth=2.0)
				# axes1[0,1].set_title("Phase for file 0", fontsize=10, fontweight='bold')
				# axes1[0,1].set_xlabel('Index', fontsize=10, fontweight = 'bold')
				# axes1[0,1].set_ylabel('Phase (rad)', fontsize=10, fontweight = 'bold')
				# axes1[0,1].text(15000,.07*phase_std,"std of phase is %1.0f mrad" %(phase_std), fontsize=10, fontweight = 'bold')
				# axes1[0,1].text(15000,.15*phase_std,"phase shift is %1.1f +/- %1.1f (%1.1f) mrad" %(meanphaseshiftforrandomfile,phase_shift_std/np.sqrt(numShots),phase_shift_std), fontsize = 10, fontweight = 'bold')
				# #axes1[0,1].text(15000,3,"slope of phase is %1.3f mrad/msmt" %(1000*phase_slope))
				# axes1[0,1].grid()
				# axes1[0,1].set_ylim(-.3*phase_std,.3*phase_std)
				#plot the amplitude across one 20000
				amp_mean = np.mean(amplitude[12000:15000])
				amp_std = np.std(amplitude[12000:15000])
				####################################PLOT OF AMPLITUDE FROM RANDOM ATOM CYCLE###############
				# axes1[0,0].plot(ch1_x[0:numMeasurementsTotal],amplitude[0:numMeasurementsTotal],'-',color='navy',label="Beatnote amplitude",linewidth=3.0)
				# axes1[0,0].axvline(x=startref,color='orange',linewidth=2.0)
				# axes1[0,0].axvline(x=stopref,color='orange',linewidth=2.0)
				# axes1[0,0].axvline(x=startMOT,color='orange',linewidth=2.0)
				# axes1[0,0].axvline(x=stopMOT,color='orange',linewidth=2.0)				
				# axes1[0,0].axvline(x=startPulsingProbe1,color='orange',linewidth=2.0)
				# axes1[0,0].axvline(x=stopPulsingProbe1,color='orange',linewidth=2.0)
				# axes1[0,0].set_title("Amplitude for file 0", fontsize=10, fontweight='bold')
				# axes1[0,0].set_xlabel('Index', fontsize=10, fontweight = 'bold')
				# axes1[0,0].set_ylabel('Amplitude (mV)', fontsize=10, fontweight = 'bold')
				# axes1[0,0].text(0,1.7*amp_mean,"amplitude mean is %1.1f +/- %1.1fmV" %(np.mean(amplitude[startMOT:stopMOT]), np.std(amplitude[startMOT:stopMOT])), fontsize=10, fontweight = 'bold')
				# axes1[0,0].text(0,1.9*amp_mean,"OD is %1.2f" %(OD_file), fontsize=10, fontweight = 'bold')
				# axes1[0,0].grid()
				# axes1[0,0].set_ylim(-1,1.1*np.max(amplitude))

				avg_digital_data = np.mean(DigitalData1_in_a_shotProbe1,0)
				####################################PLOT OF DIGITAL DATA IN SHOT FROM RANDOM ATOM CYCLE###############
				x1 = np.arange(0,len(DigitalData1_in_a_shotProbe1[0,:]))
				axes1[0,1].plot(x1,DigitalData1_in_a_shotProbe1[0,:]+.1,'-',color='navy',label="shot 1",linewidth=3.0)
				axes1[0,1].plot(x1,DigitalData1_in_a_shotProbe1[75,:]+.2,'-',color='green',label="shot 76",linewidth=3.0)
				axes1[0,1].plot(x1,DigitalData1_in_a_shotProbe1[150,:]+.3,'-',color='orange',label="shot 151",linewidth=3.0)
				axes1[0,1].plot(x1,DigitalData1_in_a_shotProbe1[225,:]+.4,'-',color='k',label="shot 226",linewidth=3.0)
				axes1[0,1].plot(x1,DigitalData1_in_a_shotProbe1[235,:]+.5,'-',color='red',label="shot 301",linewidth=3.0)
				axes1[0,1].plot(x1,DigitalData1_in_a_shotProbe1[245,:]+.6,'-',color='blue',label="shot 375",linewidth=3.0)
				axes1[0,1].plot(x1,10*avg_digital_data+2,'s-',color='orange',label="averaged shot",linewidth=3.0)
				axes1[0,1].text(0,3,"Count1 is %i"%(digitalChannel1CounterProbe1),fontsize=10, fontweight = 'bold')
				axes1[0,1].set_title("Digital Data (1) in a shot for file 2", fontsize=10, fontweight='bold')
				axes1[0,1].set_xlabel('Index', fontsize=10, fontweight = 'bold')
				axes1[0,1].set_ylabel('Value (bit)', fontsize=10, fontweight = 'bold')
				axes1[0,1].text(0,1,"digital data delayed by %i shots"%delayshift)
				axes1[0,1].grid()
				axes1[0,1].set_ylim(0,5)
				# np.savetxt(dir_main+"digital_data_example.csv",digital_data[0:numMeasurementsTotal],delimiter = ',') #Kyle, 20220113
				plt.tight_layout()

		#Averages of things per file (100 atom cycles)
		OD_dir[i]=OD_file/SegmentsPerFile
		phase_shift_dir[i]=phase_shiftProbe1_File
		phase_shift_CLICK_dir[i]=phase_shift_CLICKProbe1_File 
		phase_shift_NOCLICK_dir[i]=phase_shift_NOCLICKProbe1_File
		# amplitude_files+=amp_dir_test

		phase_shiftProbe1_File = 0
		phase_shift_CLICKProbe1_File = 0
		phase_shift_NOCLICKProbe1_File = 0
		i+=1


# amplitude_files=amplitude_files/numFiles
# final_amp=np.mean(amplitude_files,0)
# # probePulsing_OD=-2*np.log(final_amp/mean_amplitude)
# # np.savetxt(dir_main+"ODpulsing.csv",probePulsing_OD,delimiter = ',')
# np.savetxt(dir_main+"amp_pulsing.csv",final_amp,delimiter = ',')
startref = 0 
stopref = scansize
startMOT = scansize
stopMOT = 2*scansize
startPulsingProbe1 = 2*scansize+72
stopPulsingProbe1 = 2*scansize+72+numMeasurementsProbe1
amplitudeAVG = SegmentsPerFile*amplitudeAVG/numAtomCycles

axes1[0,0].plot(ch1_x[0:numMeasurementsTotal],amplitudeAVG,'-',color='navy',label="Beatnote amplitude",linewidth=3.0)
axes1[0,0].axvline(x=startref,color='orange',linewidth=2.0)
axes1[0,0].axvline(x=stopref,color='orange',linewidth=2.0)
axes1[0,0].axvline(x=startMOT,color='orange',linewidth=2.0)
axes1[0,0].axvline(x=stopMOT,color='orange',linewidth=2.0)				
axes1[0,0].axvline(x=startPulsingProbe1,color='orange',linewidth=2.0)
axes1[0,0].axvline(x=stopPulsingProbe1,color='orange',linewidth=2.0)
axes1[0,0].set_title("Amplitude for file 0", fontsize=10, fontweight='bold')
axes1[0,0].set_xlabel('Index', fontsize=10, fontweight = 'bold')
axes1[0,0].set_ylabel('Amplitude (mV)', fontsize=10, fontweight = 'bold')
axes1[0,0].set_ylim(-1,1.2*np.max(amplitudeAVG))
axes1[0,0].text(0,1.1*amp_mean,"amplitude mean is %1.1f +/- %1.1fmV" %(np.mean(amplitudeAVG[startPulsingProbe1:stopPulsingProbe1]), np.std(amplitudeAVG[startPulsingProbe1:stopPulsingProbe1])), fontsize=10, fontweight = 'bold')
# axes1[0,0].text(0,1.15*amp_mean,"OD is %1.2f" %(OD_file), fontsize=10, fontweight = 'bold')
axes1[0,0].grid()


print("%i atom cycles analyzed" %numAtomCycles)
#print("%i phase noise" %np.mean(std_dir_test))
if Spectrum == True: 
	startspectrum = 1500
	stopspectrum = 4500
	offset_spectrum = -150+25
	phaseAVG = SegmentsPerFile*(phaseAVG)/numAtomCycles
	amplitude_nomot = amplitudeAVG[startspectrum:stopspectrum]-zerovalue
	amplitude_mot = (amplitudeAVG[scansize+startspectrum-offset_spectrum:scansize+stopspectrum-offset_spectrum])-zerovalue
	phase_nomot = phaseAVG[startspectrum:stopspectrum] - np.mean(phaseAVG[startspectrum:stopspectrum])
	phase_mot = phaseAVG[scansize+startspectrum-offset_spectrum:scansize+stopspectrum-offset_spectrum] - np.mean(phaseAVG[scansize+startspectrum-offset_spectrum:scansize+stopspectrum-offset_spectrum])
	opticaldepth = -2*np.log(amplitude_mot/amplitude_nomot)
	#frequency scale (scan goes from 69.08 to 92.68, which is 23.6MHz, so through the DP that is 47.2MHz in 50us (or 3125/2 points))
	#so each point is 0.030208MHhz apart (divide by 2 if you are sampling at 16ns now)
	x_axis_spectrum = ch1_x[startspectrum:stopspectrum]*0.030208/2
	gammae = 3
	def func_lorentzian(x,a,b,center):
		return a/((b/2)**2 + (x-center)**2)
	def dispersive_lorentzian(x,a,b,center):
		return -a*(x-center)/((b/2)**2 + (x-center)**2)/4.

	center_lower =40
	center_upper = 70
	startfit = 500
	stopfit = 2500
	popt_lorentzian,pcov_popt_lorentzian = curve_fit(func_lorentzian,x_axis_spectrum[startfit:stopfit],opticaldepth[startfit:stopfit],bounds=([1e-3,1e-3,center_lower], [1e3,10,center_upper]))
	center_spectrum = popt_lorentzian[2]
	width = np.round((popt_lorentzian[1]),2)
	peakod = np.max(func_lorentzian(x_axis_spectrum,*popt_lorentzian))
	x_axis_spectrum_centered = x_axis_spectrum - center_spectrum
	###################################PLOT OF AVG PHASE OVER AN ATOM CYCLE FOR EVERY ATOM CYCLE###############		
	axes1[1,1].plot(x_axis_spectrum_centered,opticaldepth,'-',color='red',label="AVG beatnote amplitude",linewidth=3.0)
	axes1[1,1].plot(x_axis_spectrum_centered,phase_mot-phase_nomot-1,'-',color='blue',label="AVG beatnote amplitude",linewidth=3.0)
	axes1[1,1].plot(x_axis_spectrum_centered,func_lorentzian(x_axis_spectrum,*popt_lorentzian),'-',color='blue',label="Fit",linewidth=2.0)
	axes1[1,1].plot(x_axis_spectrum_centered,dispersive_lorentzian(x_axis_spectrum,*popt_lorentzian)-1,'-',color='red',label="Fit",linewidth=2.0)
	axes1[1,1].text(0,-2,"Lorentzian is %1.1FMHz wide"%width)
	axes1[1,1].text(5,1,"peak OD is %1.1f" %peakod)
	axes1[1,1].axvline(x=x_axis_spectrum[startfit]-center_spectrum,color='orange',linewidth=2.0)
	axes1[1,1].axvline(x=x_axis_spectrum[stopfit]-center_spectrum,color='orange',linewidth=2.0)
	stringname_spectrum1 = dir_main+"x_axis.csv"
	stringname_spectrum2 = dir_main+"OD.csv"
	stringname_spectrum3 = dir_main+"Phase.csv"

	#np.savetxt(stringname_spectrum1,(x_axis_spectrum_centered),delimiter = ',')
	#np.savetxt(stringname_spectrum2,(opticaldepth),delimiter = ',')
	#np.savetxt(stringname_spectrum3,(phase_mot-phase_nomot),delimiter = ',')

axes1[1,1].grid()
axes1[1,1].set_ylim(-3,3)


#-----CALCULATE THINGS--------------------------
N1_dir = DigitalChannel1Counter_dir
P11 = N1_dir/numShots/numAtomCycles
axes1[0,1].text(0,4,"P1(N1) is %1.3f (%f)"%(P11,N1_dir),fontsize=10, fontweight='bold')

averaged_amplitude_in_a_shot_for_dir_final = averaged_amplitude_in_a_shot_for_dir/numAtomCycles
averaged_phase_in_a_shot_for_dir_final = averaged_phase_in_a_shot_for_dir/numAtomCycles
averaged_phase_in_a_shot_CLICK_for_dir_final = averaged_phase_in_a_shot_CLICK_for_dir/numAtomCycles
averaged_phase_in_a_shot_NOCLICK_for_dir_final = averaged_phase_in_a_shot_NOCLICK_for_dir/numAtomCycles

x3 = np.arange(0,len(averaged_phase_in_a_shot_for_dir_final))
xdata = np.append(np.arange(start1,stop1,1),np.arange(start3,stop3,1))

averaged_phase_in_a_shot_for_dir_final_zeromean = averaged_phase_in_a_shot_for_dir_final-np.mean(averaged_phase_in_a_shot_for_dir_final)
yPhaseData = np.append(averaged_phase_in_a_shot_for_dir_final_zeromean[start1:stop1],averaged_phase_in_a_shot_for_dir_final_zeromean[start3:stop3])
m,b = fit_to_a_line(xdata,yPhaseData)
averaged_phase_in_a_shot_for_dir_final_zeromean_noslope = averaged_phase_in_a_shot_for_dir_final_zeromean-x3*m-b

averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean=averaged_phase_in_a_shot_CLICK_for_dir_final-np.mean(averaged_phase_in_a_shot_CLICK_for_dir_final)
yPhaseDataClick = np.append(averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean[start1:stop1],averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean[start3:stop3])
mClick,bClick = fit_to_a_line(xdata,yPhaseDataClick)
averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean_noslope = averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean-x3*mClick-bClick

averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean=averaged_phase_in_a_shot_NOCLICK_for_dir_final-np.mean(averaged_phase_in_a_shot_NOCLICK_for_dir_final)
yPhaseDataNOClick = np.append(averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean[start1:stop1],averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean[start3:stop3])
mNOClick,bNOClick = fit_to_a_line(xdata,yPhaseDataNOClick)
averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean_noslope = averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean-x3*mNOClick-bNOClick

average_amplitude = np.mean(averaged_amplitude_in_a_shot_for_dir_final)
amplitude_std = np.std(averaged_amplitude_in_a_shot_for_dir_final)

#Calculate STD of the mean using sums 
mean_XPS=phase_shiftProbe1_1/(numAtomCycles)
mean_square_XPS=square_phase_shift/(numAtomCycles)
phase_shift_dir_stem=1000*np.sqrt((mean_square_XPS-mean_XPS**2)/(numShots*numAtomCycles))

#Calculate STD of the CvNC using sums 
STD_phase_in_a_shot_CLICK=1000*np.sqrt(((square_phase_in_a_shot_CLICK_sum/N1_dir)-(phase_in_a_shot_CLICK_sum/N1_dir)**2)/N1_dir)
STD_phase_in_a_shot_NOCLICK=1000*np.sqrt(((square_phase_in_a_shot_NOCLICK_sum/(numShots*numAtomCycles-N1_dir))-(phase_in_a_shot_NOCLICK_sum/(numShots*numAtomCycles-N1_dir))**2)/(numShots*numAtomCycles-N1_dir))

x2 = np.arange(0,len(phase_shift_dir))
phase_shift_dir_averaged = 1000*np.mean(phase_shift_dir)
#phase_shift_dir_stem = 1000*np.std(phase_shift_dir)/np.sqrt(len(phase_shift_dir)) #phase_shift_dir is the average XPS for each file analyzed (num elements = num files)
phase_shift_dir_std = 1000*np.std(phase_shift_dir)
phase_shift_CLICK_dir_averaged = 1000*np.mean(phase_shift_CLICK_dir)
phase_shift_CLICK_dir_stem = 1000*np.std(phase_shift_CLICK_dir)/np.sqrt(len(phase_shift_CLICK_dir))
phase_shift_CLICK_dir_std = 1000*np.std(phase_shift_CLICK_dir)
phase_shift_NOCLICK_dir_averaged = 1000*np.mean(phase_shift_NOCLICK_dir)
phase_shift_NOCLICK_dir_stem = 1000*np.std(phase_shift_NOCLICK_dir)/np.sqrt(len(phase_shift_NOCLICK_dir))
phase_shift_NOCLICK_dir_std = 1000*np.std(phase_shift_NOCLICK_dir)
CvNC = phase_shift_CLICK_dir_averaged-phase_shift_NOCLICK_dir_averaged
CvNC_std = np.sqrt((phase_shift_CLICK_dir_stem)**2+(phase_shift_NOCLICK_dir_stem)**2)


###################################PLOT OF AVG PHASE SHIFT PER SHOT FOR EVERY ATOM CYCLE###############		
axes1[1,0].plot(x2,1000*phase_shift_dir,'s',color='navy',linewidth=3.0)
axes1[1,0].set_title("Phase shift in a shot for the run", fontsize=10, fontweight='bold')
axes1[1,0].set_xlabel('shot #', fontsize=10, fontweight = 'bold')
axes1[1,0].set_ylabel('Shift (mrad)', fontsize=10, fontweight = 'bold')
axes1[1,0].grid()
axes1[1,0].text(1,  -15*phase_shift_dir_std, dir_main[0:40], fontsize=10, fontweight='bold' )
axes1[1,0].text(1,  -20*phase_shift_dir_std, dir_main[40:80], fontsize=10, fontweight='bold' )
axes1[1,0].text(1,  30*phase_shift_dir_std, r'averaged phase shift is %1.3f +/- %1.3f mrad' %((phase_shift_dir_averaged), (phase_shift_dir_stem)), fontsize=10, fontweight='bold' )
axes1[1,0].text(1,  4*phase_shift_dir_std, r'Uncompensated phase shift is %1.3f +/- %1.3f (%1.2f) mrad' %(phase_shift_dir_averaged, phase_shift_dir_stem,phase_shift_dir_std), fontsize=10, fontweight='bold' )
axes1[1,0].text(1,  12*phase_shift_dir_std, r'phase shift for CLICK is %1.3f +/- %1.3f (%1.2f) mrad' %(phase_shift_CLICK_dir_averaged, phase_shift_CLICK_dir_stem,phase_shift_CLICK_dir_std), fontsize=10, fontweight='bold' )
axes1[1,0].text(1,  8*phase_shift_dir_std, r'phase shift for NO CLICK is %1.3f +/- %1.3f (%1.2f) mrad' %(phase_shift_NOCLICK_dir_averaged, phase_shift_NOCLICK_dir_stem,phase_shift_NOCLICK_dir_std), fontsize=10, fontweight='bold' )
axes1[1,0].text(1,  16*phase_shift_dir_std, r'phase shift  is %1.3f +/- %1.3f (%1.2f) mrad' %(phase_shift_dir_averaged, phase_shift_dir_stem,phase_shift_dir_std), fontsize=10, fontweight='bold' )
axes1[1,0].text(1,  20*phase_shift_dir_std, r'Click vs No click is %1.3f +/- %1.3f mrad' %(CvNC,CvNC_std), fontsize=10, fontweight='bold' )
axes1[1,0].text(1,  24*phase_shift_dir_std, r'%i measurements' %(len(phase_shift_dir)), fontsize=10, fontweight='bold' )
axes1[1,0].set_ylim(-30*phase_shift_dir_std,45*phase_shift_dir_std)
#plt.tight_layout()	


axes11ylim = 1.5*max(abs(5*np.min(averaged_phase_in_a_shot_for_dir_final_zeromean)),5*np.max(averaged_phase_in_a_shot_for_dir_final_zeromean),2e-4)
offset1=axes11ylim/3
phaseshift_otherway = 1000*(np.mean(averaged_phase_in_a_shot_for_dir_final_zeromean_noslope[start2:stop2]))
measured = phaseshift_otherway

CvNC_difference = averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean_noslope-averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean_noslope

###################################PLOT OF AVG PHASE IN A SHOT ACROSS ALL ATOM CYCLES###############		
axes1[2,1].plot(x3,averaged_phase_in_a_shot_for_dir_final_zeromean,'s-',color='orange',linewidth=1.0)
axes1[2,1].plot(x3,averaged_phase_in_a_shot_for_dir_final_zeromean_noslope,'s-',color='navy',linewidth=3.0)
axes1[2,1].plot(x3,CvNC_difference+offset1,'s-',color='navy',linewidth=3.0)
axes1[2,1].set_title("Phase avg in a shot for run", fontsize=10, fontweight='bold')
axes1[2,1].set_xlabel('Index', fontsize=10, fontweight = 'bold')
axes1[2,1].set_ylabel('Phase (rad)', fontsize=10, fontweight = 'bold')
axes1[2,1].text(1,  -.3*axes11ylim, r'Phase shifts are %1.3f +/- %1.5f mrad' %(phaseshift_otherway,phase_shift_dir_stem), fontsize=10, fontweight='bold' )
axes1[2,1].text(1,  -.5*axes11ylim, r'Phase shifts are corrected %1.3f +/- %1.3f mrad' %(measured,phase_shift_dir_stem), fontsize=10, fontweight='bold' )
axes1[2,1].text(1,  -.8*axes11ylim, r'std in shot is %1.3furad' %(1e6*np.std(averaged_phase_in_a_shot_for_dir_final_zeromean_noslope)), fontsize=10, fontweight='bold' )
axes1[2,1].axvline(x=start1,color='orange',linewidth=2.0)
axes1[2,1].axvline(x=stop1,color='orange',linewidth=2.0)
axes1[2,1].axvline(x=start2,color='orange',linewidth=2.0)
axes1[2,1].axvline(x=stop2,color='orange',linewidth=2.0)
axes1[2,1].axvline(x=start3,color='orange',linewidth=2.0)
axes1[2,1].axvline(x=stop3,color='orange',linewidth=2.0)
axes1[2,1].grid()
axes1[2,1].set_ylim(-axes11ylim,axes11ylim)
#plt.tight_layout()
stringname = dir_main+".csv"
stringname2 = stringname.replace('/','_')
stringname2 = stringname2[8:]
stringname22 = dir_main+"XPS.csv"
stringname23 = dir_main+"CvNC.csv"
stringname24 = dir_main+"SPCM.csv"
stringname25 = dir_main+"Cphase.csv"
stringname26 = dir_main+"NCphase.csv"

np.savetxt(stringname22,(averaged_phase_in_a_shot_for_dir_final_zeromean_noslope),delimiter=',')
#np.savetxt(stringname22,(averaged_phase_in_a_shot_for_dir_final),delimiter=',')
np.savetxt(stringname23,(CvNC_difference),delimiter = ',')
np.savetxt(stringname24,(avg_digital_data),delimiter=',')
np.savetxt(stringname25,(averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean_noslope),delimiter=',')
np.savetxt(stringname26,(averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean_noslope),delimiter=',')


####################################PLOT OF AVG OD FOR EACH ATOM CYCLE###############		
xx = np.arange(0,len(OD_dir))
avgOD = np.mean(OD_dir)
axes1[2,0].plot(xx,OD_dir, 's-', color = 'blue', linewidth = 1.0)
axes1[2,0].set_title("OD in each file")
axes1[2,0].set_ylabel("OD")
axes1[2,0].set_xlabel("File #")
axes1[2,0].set_ylim(0,3)
axes1[2,0].text(1,2,'average OD is %1.2f' %avgOD)
axes1[2,0].text(1,1.6, filepath[0:17])
axes1[2,0].text(1,1.4, filepath[17:100])
axes1[2,0].text(1,1.2, filepath[100:180])

stringnamepng = dir_main+".png"
plt.savefig(stringnamepng)

#save a background phase and OD
#np.savetxt("J:/Data/20190501/BackgroundData/BackgroundOD.csv",ODMOT,delimiter=',')
#np.savetxt("J:/Data/20190501/BackgroundData/BackgroundPhase.csv",phase_scan2,delimiter=',')

#this is temporary code to find the XPS as a function of time within an atom cycle after averaging all the atom cycles. 

#avg_phase_in_a_cycle=SegmentsPerFile*phase_in_a_cycle/numAtomCycles-phase_slope*ch1_x[2*scansize+72:numMeasurementsTotal]
avg_phase_in_a_cycle=SegmentsPerFile*phase_in_a_cycle/numAtomCycles
m,b= fit_to_a_line(ch1_x[2*scansize+72+35000:numMeasurementsTotal],avg_phase_in_a_cycle[35000:len(avg_phase_in_a_cycle)])
avg_phase_in_a_cycle=avg_phase_in_a_cycle-(m*ch1_x[2*scansize+72:numMeasurementsTotal]+b)
stringnamepng = dir_main+"phase_in_a_cycle.csv"
np.savetxt(stringnamepng,(avg_phase_in_a_cycle),delimiter = ',')

print(dir_main)
stoptime = time.time()
print("Program took %1.2f seconds" %(stoptime-starttime))

plt.figure(figsize=(7.2,5))
#plt.axhline(y=0, color='b', linestyle='-')
plt.xlabel(r"measurement", fontsize = 18)
plt.ylabel(r'phase', fontsize = 18)
plt.plot(np.arange(len(avg_phase_in_a_cycle)),avg_phase_in_a_cycle,'go')
stringnamepng = dir_main+"average_phase.png"
plt.savefig(stringnamepng,format='png', dpi=400)
plt.show()

# Phase_in_a_shot = avg_phase_in_a_cycle.reshape(numShots,numMeasurementsPerShot)

# #subtract off linear background
# background_i_phase = np.mean(Phase_in_a_shot[:,start1:stop1],1)
# background_f_phase = np.mean(Phase_in_a_shot[:,start3:stop3],1)
# background_phase =  (background_i_phase+background_f_phase)/2
# signal_phase = np.mean(Phase_in_a_shot[:,start2:stop2],1) #getting peak value
# phase_shift = 1000*(signal_phase - background_phase)
# average_shots=103
# phase_shift_shots=np.mean(np.reshape(phase_shift,(int(numShots/average_shots),average_shots)),1)
# shots_std=np.std(np.reshape(phase_shift,(int(numShots/average_shots),average_shots)),1)
# stringnamepng = dir_main+"XPS_shots_array.csv"
# np.savetxt(stringnamepng,(phase_shift),delimiter = ',')
# stringnamepng = dir_main+"XPS_shots_average.csv"
# np.savetxt(stringnamepng,(phase_shift_shots,shots_std),delimiter = ',')
# phase_in_shot_over_time=Phase_in_a_shot.reshape(-1,average_shots,numMeasurementsPerShot).mean(axis=1)
