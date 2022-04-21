"""
----------------------------------------------------------------
Created: Dec 13, 2021
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
v9 long shots: same as v9, but for shots that have 36x4 = 108 points. Added Dec 13, 2021 by Kyle. (updated Jan 10, 2022 to work for 576*2ns shots)
v9p1 long shots:  added capability to run program while simultaneously saving data from the VI (Vida)
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

from Spectrum_info_class import Spectrum_info_class

starttime = time.time()
mpl.rcParams.update({'font.size': 10, 'font.weight': 'bold','font.family': 'STIXGeneral',
                            'mathtext.fontset': 'stix'})


# dir_main = 'F:/Data/20210726/XPS_res_OD/10_0p25'
# dir_main = 'D:/Data/20211028/linear_phase/take2/mp15'
# dir_main = 'D:/Data/20211129/CvNC_vs_noise/211photons_600mV_rms_noise'
#dir_main = 'D:/Data/20220328/linear_phase_feature_tests/10khz_3Vpp_desync_2'
# dir_main = 'D:/Data/20220330/CvNC_bypassing_atoms_long_shots/40ns_1900'
# dir_main = 'D:/Data/20220406/CvNC_bypassing_atoms/18kHz_CvNC_40ns_1630_highcr'
dir_main = 'D:/Data/20220411/CvNC_bypassing_atoms/probe_circuit_4'
# dir_main = 'D:/Data/20220411/OD_tests/OD_test_comp_coils_off_1307'
# dir_main = 'D:/Data/20211129/phi_0_633'
# dir_main = 'D:/Data/20211104/XPS_vs_probe_power/test'
#dir_main = 'sample_data_clicks'
#dir_main = 'F:/Data/20210723/XPS_vs_probe_detuning_30ns_pulses_signal_m0p06V/0p15V'
#dir_main = 'XPS_vs_probe_detuning_30ns_pulses_signal_m0p06V/0p01V'

fileendstring = '_0.tdms'

analyze_while_taking_data = True
number_of_files = 100 # Number of files that the program expects you to eventually save when "analyze_when_taking_data" is enabled

AnalyzeDigitalDataBool = True
TemporalFilteringBool = False
ReplaceBool = False
Spectrum = True
correction_factor = 6.605 #see evernote from Mar 1, 2021 for calibration
#numShotsToCorrelate = 50
numsigma = 1
delayshift = 0#delay the SPCM's by some # of shots to compensate for a 1us delay with BNCs
factor=1 #1.70*.94, this has to be checked every day (don't know)
zerovalue = 7 #this has to be checked every day (don't know)
BackgroundPhase=0 #np.genfromtxt('backgroundphaseATOMS20190823.txt', delimiter=',')
numShots = 412 #1648/4 (do 1648/2 for 2 shots)
offsetOD = 0 #(don't know)
numMeasurementsProbe1 = int(59328)
numMeasurementsTotal =  int(72000)
numMeasurementsPerShot = int(144) #72 for 2 shots
scansize = int(6300)
phase_slope=-0.000321146 #found in calibration for 10KHz LO detuning
#phase_slope=-0.0294728 #found in calibration for 300KHz LO detuning
#phase_slope=-0.000344*10*3 #found in calibration for 10KHz LO detuning
#these have to be equidistant etc otherwise the fit approach won't match the subtraction approach
#for 200ns pulses
# shift = 1#2
# start1 = 4+shift
# stop1 = 9+shift
# start2 = 16+shift #9
# stop2 = 21+shift #10
# start3 = 28+shift #17
# stop3 = 33+shift #21

# shift = 7 #2
# start1 = -2+shift
# stop1 = 2+shift
# start2 = 9+shift #9
# stop2 = 10+shift #10
# start3 = 17+shift #17
# stop3 = 21+shift #21

# # for 50ns Gaussian pulses
# shift = 21 #2
# start1 = 0+shift
# stop1 = 10+shift
# start2 = 20+shift #9
# stop2 = 21+shift #10
# start3 = 31+shift #17
# stop3 = 41+shift #21

# for 4x10ns Gaussian pulses and 4 shots
shift = 0 #2
start1 = 20+shift
stop1 = 40+shift
start2 = 60+shift #9
stop2 = 61+shift #10
start3 = 81+shift #17
stop3 = 101+shift #21

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

def dec_to_bin(x):
	m=8
	#binary_matrix=(((x[:,None].astype(int) & (1 << np.arange(m)))) > 0).astype(int)
	binary_matrix=(((x[:,None] & (1 << np.arange(m)))) > 0).astype(int)
	y1 = binary_matrix[:,1]
	y5 = binary_matrix[:,5]
	return(y1,y5)

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
	interleaved = tdms_file.object('Raw Data','Channel 0') #c0s0,c1s0,c0s1,c1s1,c0s2,c1s2, etc (c=channel, s=sample)
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

	# find the phase and unwrap it
	phase_wrapped =  np.zeros(len(Q))
	I_is_zero = np.where(I == 0)[0]
	phase_wrapped[I_is_zero] = np.pi/2
	left_to_change = np.where(phase_wrapped == 0)[0]
	phase_wrapped[left_to_change] = np.arctan(Q[left_to_change]/I[left_to_change])
	phase = np.unwrap(2*phase_wrapped,np.pi)/2

	#Warnings
	if len(ch1_x) != numMeasurementsTotal*file_info.num_segments:
		print('Warning, trace is the wrong length')
	return(ch1_x, sixteenBitIntegerTomV(amplitude,file_info.ch0_voltage_range_number), phase,digital_data, file_info)


def Analyze(amplitude, phase, digitaldata1, numShots, numMeasurements, numMeasurementsPerShot,numsigma):
	#take 27000 long array and make it into 375, 72 long arrays
	if len(amplitude) != numMeasurements:
		print('Warning, array is the wrong length')
	Phase_in_a_shot = phase.reshape(numShots,numMeasurementsPerShot)[:206,]	# 1648 msmts in total 206 is half 412 (so analyzing only half of the cycle)
	Amplitude_in_a_shot = amplitude.reshape(numShots,numMeasurementsPerShot)[:206,]
	
	#subtract off linear background
	background_i_phase = np.mean(Phase_in_a_shot[:,start1:stop1],1)
	background_f_phase = np.mean(Phase_in_a_shot[:,start3:stop3],1)
	background_phase =  (background_i_phase+background_f_phase)/2
	signal_phase = np.mean(Phase_in_a_shot[:,start2:stop2],1) #getting peak value
	phase_shift = signal_phase - background_phase

	background_i_amp = np.mean(Amplitude_in_a_shot[:,start1:stop1],1)
	background_f_amp = np.mean(Amplitude_in_a_shot[:,start3:stop3],1)
	background_amp = (background_f_amp+background_i_amp)/2.0
	signal_amp = np.mean(Amplitude_in_a_shot[:,start2:stop2],1)
	amplitude_shift = signal_amp - background_amp

	if AnalyzeDigitalDataBool == True:
		DigitalData_in_a_shot = digitaldata1.reshape(numShots,numMeasurementsPerShot)[:206,]
		#This is done to only count the arrival time bin of a click 
		b=np.zeros_like(DigitalData_in_a_shot)
		b[np.arange(len(DigitalData_in_a_shot)), DigitalData_in_a_shot.argmax(1)] = 1 #python doesn't know what to do when all the elements are equal and returns the first index
		b[:,0]=0 #I do this to set that first index back to zero in the rows that only contain zeros, the other rows don't have a problem
		DigitalData1_in_a_shot=b
		if TemporalFilteringBool == True:
			todelete = np.concatenate((np.arange(0,start2-8,1),np.arange(stop2-4,numMeasurementsPerShot,1)))
			DigitalData1_in_a_shot = np.delete(DigitalData1_in_a_shot,todelete,axis=1) #this is for temporally filtering the SPCMs
		DigitalData1 = np.roll(np.any(DigitalData1_in_a_shot,1),delayshift)
		digitalChannel1Counter=np.sum(DigitalData1)

		if np.sum(DigitalData1)>0:
			Phase_in_a_shot_CLICK = Phase_in_a_shot[DigitalData1]
			Phase_in_a_shot_NOCLICK = Phase_in_a_shot[np.logical_not(DigitalData1)]
			Amplitude_in_a_shot_CLICK = Amplitude_in_a_shot[DigitalData1]
			Amplitude_in_a_shot_NOCLICK = Amplitude_in_a_shot[np.logical_not(DigitalData1)]

			background_i_phase_CLICK = np.mean(Phase_in_a_shot_CLICK[:,start1:stop1],1)
			background_f_phase_CLICK = np.mean(Phase_in_a_shot_CLICK[:,start3:stop3],1)
			background_phase_CLICK =  (background_i_phase_CLICK+background_f_phase_CLICK)/2
			signal_phase_CLICK = np.mean(Phase_in_a_shot_CLICK[:,start2:stop2],1)
			phase_shift_CLICK = signal_phase_CLICK - background_phase_CLICK
	
			background_i_amp_CLICK = np.mean(Amplitude_in_a_shot_CLICK[:,start1:stop1],1)
			background_f_amp_CLICK = np.mean(Amplitude_in_a_shot_CLICK[:,start3:stop3],1)
			background_amp_CLICK = (background_f_amp_CLICK+background_i_amp_CLICK)/2.0
			signal_amp_CLICK = np.mean(Amplitude_in_a_shot_CLICK[:,start2:stop2],1)
			amplitude_shift_CLICK = signal_amp_CLICK - background_amp_CLICK
			
			background_i_phase_NOCLICK = np.mean(Phase_in_a_shot_NOCLICK[:,start1:stop1],1)
			background_f_phase_NOCLICK = np.mean(Phase_in_a_shot_NOCLICK[:,start3:stop3],1)
			background_phase_NOCLICK =  (background_i_phase_NOCLICK+background_f_phase_NOCLICK)/2
			signal_phase_NOCLICK = np.mean(Phase_in_a_shot_NOCLICK[:,start2:stop2],1)
			phase_shift_NOCLICK = signal_phase_NOCLICK - background_phase_NOCLICK
	
			background_i_amp_NOCLICK = np.mean(Amplitude_in_a_shot_NOCLICK[:,start1:stop1],1)
			background_f_amp_NOCLICK = np.mean(Amplitude_in_a_shot_NOCLICK[:,start3:stop3],1)
			background_amp_NOCLICK = (background_f_amp_NOCLICK+background_i_amp_NOCLICK)/2.0
			signal_amp_NOCLICK = np.mean(Amplitude_in_a_shot_NOCLICK[:,start2:stop2],1)
			amplitude_shift_NOCLICK = signal_amp_NOCLICK - background_amp_NOCLICK



		else:
			phase_shift_CLICK=phase_shift
			amplitude_shift_CLICK=amplitude_shift
			phase_shift_NOCLICK=phase_shift
			amplitude_shift_NOCLICK=amplitude_shift
			# Phase_in_a_shot_CLICK=np.zeros((900,18))
			# Phase_in_a_shot_NOCLICK=np.zeros((900,18))
			# Amplitude_in_a_shot_CLICK=np.zeros((900,18))
			# Amplitude_in_a_shot_NOCLICK=np.zeros((900,18))
			Phase_in_a_shot_CLICK=np.zeros((numShots,numMeasurementsPerShot))
			Phase_in_a_shot_NOCLICK=np.zeros((numShots,numMeasurementsPerShot))
			Amplitude_in_a_shot_CLICK=np.zeros((numShots,numMeasurementsPerShot))
			Amplitude_in_a_shot_NOCLICK=np.zeros((numShots,numMeasurementsPerShot))

	else:
		DigitalData1=np.zeros(numShots)
		DigitalData1_in_a_shot=np.zeros((numShots,numMeasurementsPerShot))
		digitalChannel1Counter=0
		phase_shift_CLICK=phase_shift
		amplitude_shift_CLICK=amplitude_shift
		phase_shift_NOCLICK=phase_shift
		amplitude_shift_NOCLICK=amplitude_shift
		Phase_in_a_shot_CLICK=np.zeros((numShots,numMeasurementsPerShot))
		Phase_in_a_shot_NOCLICK = np.zeros((numShots,numMeasurementsPerShot))
		Amplitude_in_a_shot_CLICK = np.zeros((numShots,numMeasurementsPerShot))
		Amplitude_in_a_shot_NOCLICK = np.zeros((numShots,numMeasurementsPerShot))
	

	# index,phase_shift_trunc = g2analysis(phase_shift,numsigma)
	# DigitalData1_trunc = DigitalData1[index]
	# digitalChannel1Counter_trunc=np.sum(DigitalData1_trunc)

	return(Phase_in_a_shot, Amplitude_in_a_shot, Phase_in_a_shot_CLICK,Phase_in_a_shot_NOCLICK,DigitalData1_in_a_shot, 
		phase_shift,amplitude_shift,phase_shift_CLICK,amplitude_shift_CLICK,phase_shift_NOCLICK,amplitude_shift_NOCLICK,
		digitalChannel1Counter)

def fit_to_a_line(x,y):
	def func(x,m,b):
		return m*x + b
	popt_func,pcov_func = curve_fit(func,x,y) #bounds = ([-.00031,-2],[-0.0001,1])
	return(popt_func[0],popt_func[1])

#preparing some variables which will be filled with info from all the files
DigitalChannel1Counter_dir = 0
averaged_amplitude_in_a_shot_for_dir = 0
averaged_phase_in_a_shot_for_dir = 0
averaged_phase_in_a_shot_CLICK_for_dir = 0
averaged_phase_in_a_shot_NOCLICK_for_dir = 0
phase_shiftProbe1_File=0
square_phase_shift=0
phase_shiftProbe1_1=0
phase_shift_CLICKProbe1_File=0
phase_shift_NOCLICKProbe1_File=0

amplitudeMOT = np.zeros(scansize)
phaseMOT = np.zeros(scansize)
amplituderef = np.zeros(scansize)
phaseref = np.zeros(scansize)
amplitudeEIT = np.zeros(scansize)
phaseEIT = np.zeros(scansize)
phaseAVG = np.zeros(numMeasurementsTotal)
amplitude_files=np.zeros([100,numMeasurementsProbe1])
numAtomCycles = 0
amplitudeAVG=0
phaseAVG = 0
phase_in_a_cycle=0
DigitalData1_cycles=0
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
	for fs in range(0,number_of_files):
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
			print("test_{}".format(fs))				

		#in each iteration of the loop do stuff to a different file. 
		ch1_x, amplitude, phase,digital_data, file_info  =  Load_Data(filepath) 
		SegmentsPerFile = file_info.num_segments #get num segments from file info
		#std_dir_test = np.zeros(SegmentsPerFile)
		amp_dir_test=np.zeros([SegmentsPerFile,numMeasurementsProbe1])
		if ReplaceBool == True:
			probability = .2/numMeasurementsPerShot
			digitaldatalength = len(digital_data)
			digital_data = np.random.binomial(1,probability,len(digital_data))#use this if you want a random set of digital data
		if Spectrum == True:
			amplitudeAVG += np.mean(amplitude.reshape(SegmentsPerFile,numMeasurementsTotal),0)[0:scansize*2]
			phaseAVG += np.mean(phase.reshape(SegmentsPerFile,numMeasurementsTotal),0)[0:scansize*2] 
			phase_in_a_cycle+=np.mean(phase.reshape(SegmentsPerFile,numMeasurementsTotal),0)[(2*scansize+72):numMeasurementsTotal]
		OD_file=0
		for k in range(SegmentsPerFile):
			startref = 0 + numMeasurementsTotal*k
			stopref = scansize+ numMeasurementsTotal*k
			startMOT = scansize+ numMeasurementsTotal*k
			stopMOT = 2*scansize+ numMeasurementsTotal*k
			startEIT = 2*scansize+ numMeasurementsTotal*k
			stopEIT = 3*scansize+ numMeasurementsTotal*k
			startPulsingProbe1 = 2*scansize+72+numMeasurementsTotal*k
			stopPulsingProbe1 = 2*scansize+72+numMeasurementsProbe1+ numMeasurementsTotal*k
			phase_slope,phase_offset= fit_to_a_line(ch1_x[startPulsingProbe1:stopPulsingProbe1],phase[startPulsingProbe1:stopPulsingProbe1])
			amplituderef = amplitude[startref:stopref]
			ch1PulsingProbe1_x = ch1_x[startPulsingProbe1:stopPulsingProbe1]
			amplitudePulsingProbe1 = amplitude[startPulsingProbe1:stopPulsingProbe1]
			phasePulsingProbe1 = phase[startPulsingProbe1:stopPulsingProbe1]-phase_slope*ch1_x[startPulsingProbe1:stopPulsingProbe1]
			#phasePulsingProbe1 = phase[startPulsingProbe1:stopPulsingProbe1]
			phase_std2 = 1000*np.std(phasePulsingProbe1[10000:10100]-(phase_slope*ch1_x[startPulsingProbe1+10000:startPulsingProbe1+10100])) #calculate the phase noise for 100 points subtracting the slope
			#std_dir_test[k]=phase_std2
			digitaldatapulsingProbe1 = digital_data[startPulsingProbe1:stopPulsingProbe1]
			#Feed the data into the Analyze function. 
			Phase_in_a_shotProbe1, Amplitude_in_a_shotProbe1, Phase_in_a_shot_CLICKProbe1,Phase_in_a_shot_NOCLICKProbe1,DigitalData1_in_a_shotProbe1, phase_shiftProbe1,amplitude_shiftProbe1,phase_shift_CLICKProbe1,amplitude_shift_CLICKProbe1,phase_shift_NOCLICKProbe1,amplitude_shift_NOCLICKProbe1,digitalChannel1CounterProbe1 = Analyze(amplitudePulsingProbe1,phasePulsingProbe1,digitaldatapulsingProbe1, numShots, numMeasurementsProbe1, numMeasurementsPerShot,numsigma)
			#what do we want to keep track of across all ten files?
			mean_amplitude = np.mean(amplitude[0:700])-zerovalue
			mean_amplitude2 = np.mean(amplitudePulsingProbe1[300:2000])-zerovalue
			OD_file += -2*np.log(mean_amplitude2/mean_amplitude)
			averaged_amplitude_in_a_shot_for_dir += np.mean(Amplitude_in_a_shotProbe1,0)
			averaged_phase_in_a_shot_for_dir += np.mean(Phase_in_a_shotProbe1,0)
			averaged_phase_in_a_shot_CLICK_for_dir+=np.mean(Phase_in_a_shot_CLICKProbe1,0)
			averaged_phase_in_a_shot_NOCLICK_for_dir+=np.mean(Phase_in_a_shot_NOCLICKProbe1,0)
			DigitalChannel1Counter_dir += digitalChannel1CounterProbe1
			DigitalData1_cycles+=DigitalData1_in_a_shotProbe1
			amp_dir_test[k]=amplitudePulsingProbe1
			numAtomCycles += 1

			phase_shiftProbe1_1+=np.sum(phase_shiftProbe1)/numShots
			square_phase_shift+=np.sum(phase_shiftProbe1**2)/numShots
			phase_shiftProbe1_File+=np.mean(phase_shiftProbe1)/SegmentsPerFile		# A variable
			phase_shift_CLICKProbe1_File+=np.mean(phase_shift_CLICKProbe1)/SegmentsPerFile #thats the average phase shift in an atom cycle for every atom cycle
			phase_shift_NOCLICKProbe1_File+=np.mean(phase_shift_NOCLICKProbe1)/SegmentsPerFile

			
			#now pick a random file to make some plots of so we can sanity check everything. 
			if filepath.endswith(fileendstring) and k == 0:
			#if 1000*np.max(np.mean(phase_shift))>10:
				print("This file analyzed")
				phase_std = 1000*np.std(phase[20000:21000])
				phase_mean = np.mean(phase[12000:13000])
				phase_shift_std = 1000*np.std(phase_shiftProbe1)
				meanphaseshiftforrandomfile = 1000*np.mean(phase_shiftProbe1)
				fig1,axes1 = plt.subplots(3,3,figsize=(15,8))
				#plot the phase across one file
				####################################PLOT OF PHASE FROM RANDOM ATOM CYCLE############### -phase_mean
				axes1[0,1].plot(ch1_x[0:numMeasurementsTotal],phase[0:numMeasurementsTotal]-(phase_slope*ch1_x[0:numMeasurementsTotal]),'-',color='navy',label="Phase",linewidth=3.0)
				#axes1[0,1].plot(ch1_x[0:36000],phase_fit,'-',color='green',label="fit",linewidth=3.0)				
				#axes1[0,1].plot(ch1_x[0:36000],phase[0:36000]-phase_fit,'-',color='green',label="fit",linewidth=3.0)				
				axes1[0,1].axvline(x=startref,color='orange',linewidth=2.0)
				axes1[0,1].axvline(x=stopref,color='orange',linewidth=2.0)
				axes1[0,1].axvline(x=startMOT,color='orange',linewidth=2.0)
				axes1[0,1].axvline(x=stopMOT,color='orange',linewidth=2.0)				
				axes1[0,1].axvline(x=startEIT,color='orange',linewidth=2.0)
				axes1[0,1].axvline(x=stopEIT,color='orange',linewidth=2.0)
				axes1[0,1].axvline(x=startPulsingProbe1,color='orange',linewidth=2.0)
				axes1[0,1].axvline(x=stopPulsingProbe1,color='orange',linewidth=2.0)
				axes1[0,1].set_title("Phase for file 2", fontsize=10, fontweight='bold')
				axes1[0,1].set_xlabel('Index', fontsize=10, fontweight = 'bold')
				axes1[0,1].set_ylabel('Phase (rad)', fontsize=10, fontweight = 'bold')
				axes1[0,1].text(15000,.07*phase_std,"std of phase is %1.0f mrad" %(phase_std), fontsize=10, fontweight = 'bold')
				axes1[0,1].text(15000,.15*phase_std,"phase shift is %1.1f +/- %1.1f (%1.1f) mrad" %(meanphaseshiftforrandomfile,phase_shift_std/np.sqrt(numShots),phase_shift_std), fontsize = 10, fontweight = 'bold')
				#axes1[0,1].text(15000,3,"slope of phase is %1.3f mrad/msmt" %(1000*phase_slope))					
				axes1[0,1].text(15000,-2,'test_{}'.format(fs),fontsize=10, fontweight = 'bold')
				axes1[0,1].grid()
				axes1[0,1].legend(loc='upper left', shadow=True,fontsize='10')
				#axes1[0,1].set_ylim(-3,3)
				axes1[0,1].set_ylim(-.5*phase_std,.5*phase_std)
				#plt.tight_layout()	
				#plot the amplitude across one 20000
				amp_mean = np.mean(amplitude[12000:15000])
				amp_std = np.std(amplitude[12000:15000])
				####################################PLOT OF AMPLITUDE FROM RANDOM ATOM CYCLE###############
				axes1[0,0].plot(ch1_x[0:numMeasurementsTotal],amplitude[0:numMeasurementsTotal],'-',color='navy',label="Beatnote amplitude",linewidth=3.0)
				axes1[0,0].axvline(x=startref,color='orange',linewidth=2.0)
				axes1[0,0].axvline(x=stopref,color='orange',linewidth=2.0)
				axes1[0,0].axvline(x=startMOT,color='orange',linewidth=2.0)
				axes1[0,0].axvline(x=stopMOT,color='orange',linewidth=2.0)				
				axes1[0,0].axvline(x=startEIT,color='orange',linewidth=2.0)
				axes1[0,0].axvline(x=stopEIT,color='orange',linewidth=2.0)
				axes1[0,0].axvline(x=startPulsingProbe1,color='orange',linewidth=2.0)
				axes1[0,0].axvline(x=stopPulsingProbe1,color='orange',linewidth=2.0)

				axes1[0,0].set_title("Amplitude for file 1", fontsize=10, fontweight='bold')
				axes1[0,0].set_xlabel('Index', fontsize=10, fontweight = 'bold')
				axes1[0,0].set_ylabel('Amplitude (mV)', fontsize=10, fontweight = 'bold')
				axes1[0,0].text(0,1.3*amp_mean,"amplitude mean is %1.1f +/- %1.1fmV" %(np.mean(amplitude[startMOT:stopMOT]), np.std(amplitude[startMOT:stopMOT])), fontsize=10, fontweight = 'bold')
				axes1[0,0].text(0,1.5*amp_mean,"OD is %1.2f" %(OD_file), fontsize=10, fontweight = 'bold')
				axes1[0,0].grid()
				#axes1[0,0].legend(loc='lower right', shadow=True,fontsize='10')
				axes1[0,0].set_ylim(-1,1.5*np.max(amplitude))
				#plt.tight_layout()	
				#plot the digital data across one file for 4 channels
				#print(len(Digital0))
				####################################PLOT OF DIGITAL DATA FROM RANDOM ATOM CYCLE###############
				axes1[2,1].plot(ch1_x[0:numMeasurementsTotal],digital_data[0:numMeasurementsTotal],'-',color='black',label="5",linewidth=3.0)
				axes1[2,1].set_title("Digital Data for file 2", fontsize=10, fontweight='bold')
				axes1[2,1].set_xlabel('Index', fontsize=10, fontweight = 'bold')
				axes1[2,1].set_ylabel('Digital Data (bits)', fontsize=10, fontweight = 'bold')
				axes1[2,1].grid()
				axes1[2,1].legend(loc='upper right', shadow=True,fontsize='10')
				axes1[2,1].set_ylim(0,2)
				axes1[2,1].text(0,1.0,"start1 is %1.1f"%start1,fontsize=10, fontweight = 'bold')
				axes1[2,1].text(0,1.1,"stop1 is %1.1f"%stop1,fontsize=10, fontweight = 'bold')
				axes1[2,1].text(0,1.2,"start2 is %1.1f"%start2,fontsize=10, fontweight = 'bold')
				axes1[2,1].text(0,1.3,"stop2 is %1.1f"%stop2,fontsize=10, fontweight = 'bold')
				axes1[2,1].text(0,1.4,"start3 is %1.1f"%start3,fontsize=10, fontweight = 'bold')
				axes1[2,1].text(0,1.5,"stop3 is %1.1f"%stop3,fontsize=10, fontweight = 'bold')
				avg_digital_data = np.mean(DigitalData1_in_a_shotProbe1,0)
				####################################PLOT OF DIGITAL DATA IN SHOT FROM RANDOM ATOM CYCLE###############
				x1 = np.arange(0,len(DigitalData1_in_a_shotProbe1[0,:]))
				axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[0,:]+.1,'-',color='navy',label="shot 1",linewidth=3.0)
				axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[75,:]+.2,'-',color='green',label="shot 76",linewidth=3.0)
				axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[150,:]+.3,'-',color='orange',label="shot 151",linewidth=3.0)
				# axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[225,:]+.4,'-',color='k',label="shot 226",linewidth=3.0)
				# axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[235,:]+.5,'-',color='red',label="shot 301",linewidth=3.0)
				# axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[245,:]+.6,'-',color='blue',label="shot 375",linewidth=3.0)
				axes1[2,2].plot(x1,10*avg_digital_data+2,'s-',color='orange',label="averaged shot",linewidth=3.0)
				axes1[2,2].text(0,3,"Count1 is %i"%(digitalChannel1CounterProbe1),fontsize=10, fontweight = 'bold')
				axes1[2,2].set_title("Digital Data (1) in a shot for file 2", fontsize=10, fontweight='bold')
				axes1[2,2].set_xlabel('Index', fontsize=10, fontweight = 'bold')
				axes1[2,2].set_ylabel('Value (bit)', fontsize=10, fontweight = 'bold')
				axes1[2,2].text(0,1,"digital data delayed by %i shots"%delayshift)
				axes1[2,2].grid()
				axes1[2,2].set_ylim(0,5)
				plt.tight_layout()

		#Averages of things per file (100 atom cycles)
		OD_dir[i]=OD_file/SegmentsPerFile
		phase_shift_dir[i]=phase_shiftProbe1_File
		phase_shift_CLICK_dir[i]=phase_shift_CLICKProbe1_File 
		phase_shift_NOCLICK_dir[i]=phase_shift_NOCLICKProbe1_File
		amplitude_files+=amp_dir_test

		phase_shiftProbe1_File = 0
		phase_shift_CLICKProbe1_File = 0
		phase_shift_NOCLICKProbe1_File = 0
		i+=1

#if analyze_while_taking_data == False:
else:
	print("Running normal code")
	for root, dirs,files in os.walk(dir_main):
		numFiles=len(files)/2. 
		phase_shift_dir = np.zeros(int(numFiles))	
		phase_shift_CLICK_dir = np.zeros(int(numFiles))								
		phase_shift_NOCLICK_dir = np.zeros(int(numFiles))													
		OD_dir = np.zeros(int(numFiles))														
		i=0
		for file in files: #just iterate through all the files in the directory
			if file.endswith('tdms'):
				print(file)
				filepath = os.path.join(root,file) ################################################################################################  Uses root, files --> os.walk
				#in each iteration of the loop do stuff to a different file. 
				ch1_x, amplitude, phase,digital_data, file_info  =  Load_Data(filepath) ###########################################################  Uses filepath --> root, file --> os.walk
				SegmentsPerFile = file_info.num_segments #get num segments from file info
				#std_dir_test = np.zeros(SegmentsPerFile)
				amp_dir_test=np.zeros([SegmentsPerFile,numMeasurementsProbe1])
				if ReplaceBool == True:
					probability = .2/numMeasurementsPerShot
					digitaldatalength = len(digital_data)
					digital_data = np.random.binomial(1,probability,len(digital_data))#use this if you want a random set of digital data
				if Spectrum == True:
					amplitudeAVG += np.mean(amplitude.reshape(SegmentsPerFile,numMeasurementsTotal),0)[0:scansize*2]
					phaseAVG += np.mean(phase.reshape(SegmentsPerFile,numMeasurementsTotal),0)[0:scansize*2] 
					phase_in_a_cycle+=np.mean(phase.reshape(SegmentsPerFile,numMeasurementsTotal),0)[(2*scansize+72):numMeasurementsTotal]
				OD_file=0
				for k in range(SegmentsPerFile):
					startref = 0 + numMeasurementsTotal*k
					stopref = scansize+ numMeasurementsTotal*k
					startMOT = scansize+ numMeasurementsTotal*k
					stopMOT = 2*scansize+ numMeasurementsTotal*k
					startEIT = 2*scansize+ numMeasurementsTotal*k
					stopEIT = 3*scansize+ numMeasurementsTotal*k
					startPulsingProbe1 = 2*scansize+72+numMeasurementsTotal*k
					stopPulsingProbe1 = 2*scansize+72+numMeasurementsProbe1+ numMeasurementsTotal*k
					phase_slope,phase_offset= fit_to_a_line(ch1_x[startPulsingProbe1:stopPulsingProbe1],phase[startPulsingProbe1:stopPulsingProbe1])
					amplituderef = amplitude[startref:stopref]
					ch1PulsingProbe1_x = ch1_x[startPulsingProbe1:stopPulsingProbe1]
					amplitudePulsingProbe1 = amplitude[startPulsingProbe1:stopPulsingProbe1]
					phasePulsingProbe1 = phase[startPulsingProbe1:stopPulsingProbe1]-phase_slope*ch1_x[startPulsingProbe1:stopPulsingProbe1]
					#phasePulsingProbe1 = phase[startPulsingProbe1:stopPulsingProbe1]
					phase_std2 = 1000*np.std(phasePulsingProbe1[10000:10100]-(phase_slope*ch1_x[startPulsingProbe1+10000:startPulsingProbe1+10100])) #calculate the phase noise for 100 points subtracting the slope
					#std_dir_test[k]=phase_std2
					digitaldatapulsingProbe1 = digital_data[startPulsingProbe1:stopPulsingProbe1]
					#Feed the data into the Analyze function. 
					Phase_in_a_shotProbe1, Amplitude_in_a_shotProbe1, Phase_in_a_shot_CLICKProbe1,Phase_in_a_shot_NOCLICKProbe1,DigitalData1_in_a_shotProbe1, phase_shiftProbe1,amplitude_shiftProbe1,phase_shift_CLICKProbe1,amplitude_shift_CLICKProbe1,phase_shift_NOCLICKProbe1,amplitude_shift_NOCLICKProbe1,digitalChannel1CounterProbe1 = Analyze(amplitudePulsingProbe1,phasePulsingProbe1,digitaldatapulsingProbe1, numShots, numMeasurementsProbe1, numMeasurementsPerShot,numsigma)
					#what do we want to keep track of across all ten files?
					mean_amplitude = np.mean(amplitude[0:700])-zerovalue
					mean_amplitude2 = np.mean(amplitudePulsingProbe1[300:2000])-zerovalue
					OD_file += -2*np.log(mean_amplitude2/mean_amplitude)
					averaged_amplitude_in_a_shot_for_dir += np.mean(Amplitude_in_a_shotProbe1,0)
					averaged_phase_in_a_shot_for_dir += np.mean(Phase_in_a_shotProbe1,0)
					averaged_phase_in_a_shot_CLICK_for_dir+=np.mean(Phase_in_a_shot_CLICKProbe1,0)
					averaged_phase_in_a_shot_NOCLICK_for_dir+=np.mean(Phase_in_a_shot_NOCLICKProbe1,0)
					DigitalChannel1Counter_dir += digitalChannel1CounterProbe1
					DigitalData1_cycles+=DigitalData1_in_a_shotProbe1
					amp_dir_test[k]=amplitudePulsingProbe1
					numAtomCycles += 1

					phase_shiftProbe1_1+=np.sum(phase_shiftProbe1)/numShots
					square_phase_shift+=np.sum(phase_shiftProbe1**2)/numShots
					phase_shiftProbe1_File+=np.mean(phase_shiftProbe1)/SegmentsPerFile		# A variable
					phase_shift_CLICKProbe1_File+=np.mean(phase_shift_CLICKProbe1)/SegmentsPerFile #thats the average phase shift in an atom cycle for every atom cycle
					phase_shift_NOCLICKProbe1_File+=np.mean(phase_shift_NOCLICKProbe1)/SegmentsPerFile

					
					#now pick a random file to make some plots of so we can sanity check everything. 
					if file.endswith(fileendstring) and k == 0:
					#if 1000*np.max(np.mean(phase_shift))>10:
						print("This file analyzed")
						phase_std = 1000*np.std(phase[20000:21000])
						phase_mean = np.mean(phase[12000:13000])
						phase_shift_std = 1000*np.std(phase_shiftProbe1)
						meanphaseshiftforrandomfile = 1000*np.mean(phase_shiftProbe1)
						fig1,axes1 = plt.subplots(3,3,figsize=(15,8))
						#plot the phase across one file
						####################################PLOT OF PHASE FROM RANDOM ATOM CYCLE############### -phase_mean
						axes1[0,1].plot(ch1_x[0:numMeasurementsTotal],phase[0:numMeasurementsTotal]-(phase_slope*ch1_x[0:numMeasurementsTotal]),'-',color='navy',label="Phase",linewidth=3.0)
						#axes1[0,1].plot(ch1_x[0:36000],phase_fit,'-',color='green',label="fit",linewidth=3.0)				
						#axes1[0,1].plot(ch1_x[0:36000],phase[0:36000]-phase_fit,'-',color='green',label="fit",linewidth=3.0)				
						axes1[0,1].axvline(x=startref,color='orange',linewidth=2.0)
						axes1[0,1].axvline(x=stopref,color='orange',linewidth=2.0)
						axes1[0,1].axvline(x=startMOT,color='orange',linewidth=2.0)
						axes1[0,1].axvline(x=stopMOT,color='orange',linewidth=2.0)				
						axes1[0,1].axvline(x=startEIT,color='orange',linewidth=2.0)
						axes1[0,1].axvline(x=stopEIT,color='orange',linewidth=2.0)
						axes1[0,1].axvline(x=startPulsingProbe1,color='orange',linewidth=2.0)
						axes1[0,1].axvline(x=stopPulsingProbe1,color='orange',linewidth=2.0)
						axes1[0,1].set_title("Phase for file 2", fontsize=10, fontweight='bold')
						axes1[0,1].set_xlabel('Index', fontsize=10, fontweight = 'bold')
						axes1[0,1].set_ylabel('Phase (rad)', fontsize=10, fontweight = 'bold')
						axes1[0,1].text(15000,.07*phase_std,"std of phase is %1.0f mrad" %(phase_std), fontsize=10, fontweight = 'bold')
						axes1[0,1].text(15000,.15*phase_std,"phase shift is %1.1f +/- %1.1f (%1.1f) mrad" %(meanphaseshiftforrandomfile,phase_shift_std/np.sqrt(numShots),phase_shift_std), fontsize = 10, fontweight = 'bold')
						#axes1[0,1].text(15000,3,"slope of phase is %1.3f mrad/msmt" %(1000*phase_slope))					
						axes1[0,1].text(15000,-2,file,fontsize=10, fontweight = 'bold')
						axes1[0,1].grid()
						axes1[0,1].legend(loc='upper left', shadow=True,fontsize='10')
						#axes1[0,1].set_ylim(-3,3)
						axes1[0,1].set_ylim(-.5*phase_std,.5*phase_std)
						#plt.tight_layout()	
						#plot the amplitude across one 20000
						amp_mean = np.mean(amplitude[12000:15000])
						amp_std = np.std(amplitude[12000:15000])
						####################################PLOT OF AMPLITUDE FROM RANDOM ATOM CYCLE###############
						axes1[0,0].plot(ch1_x[0:numMeasurementsTotal],amplitude[0:numMeasurementsTotal],'-',color='navy',label="Beatnote amplitude",linewidth=3.0)
						axes1[0,0].axvline(x=startref,color='orange',linewidth=2.0)
						axes1[0,0].axvline(x=stopref,color='orange',linewidth=2.0)
						axes1[0,0].axvline(x=startMOT,color='orange',linewidth=2.0)
						axes1[0,0].axvline(x=stopMOT,color='orange',linewidth=2.0)				
						axes1[0,0].axvline(x=startEIT,color='orange',linewidth=2.0)
						axes1[0,0].axvline(x=stopEIT,color='orange',linewidth=2.0)
						axes1[0,0].axvline(x=startPulsingProbe1,color='orange',linewidth=2.0)
						axes1[0,0].axvline(x=stopPulsingProbe1,color='orange',linewidth=2.0)

						axes1[0,0].set_title("Amplitude for file 1", fontsize=10, fontweight='bold')
						axes1[0,0].set_xlabel('Index', fontsize=10, fontweight = 'bold')
						axes1[0,0].set_ylabel('Amplitude (mV)', fontsize=10, fontweight = 'bold')
						axes1[0,0].text(0,1.3*amp_mean,"amplitude mean is %1.1f +/- %1.1fmV" %(np.mean(amplitude[startMOT:stopMOT]), np.std(amplitude[startMOT:stopMOT])), fontsize=10, fontweight = 'bold')
						axes1[0,0].text(0,1.5*amp_mean,"OD is %1.2f" %(OD_file), fontsize=10, fontweight = 'bold')
						axes1[0,0].grid()
						#axes1[0,0].legend(loc='lower right', shadow=True,fontsize='10')
						axes1[0,0].set_ylim(-1,1.5*np.max(amplitude))
						#plt.tight_layout()	
						#plot the digital data across one file for 4 channels
						#print(len(Digital0))
						####################################PLOT OF DIGITAL DATA FROM RANDOM ATOM CYCLE###############
						axes1[2,1].plot(ch1_x[0:numMeasurementsTotal],digital_data[0:numMeasurementsTotal],'-',color='black',label="5",linewidth=3.0)
						axes1[2,1].set_title("Digital Data for file 2", fontsize=10, fontweight='bold')
						axes1[2,1].set_xlabel('Index', fontsize=10, fontweight = 'bold')
						axes1[2,1].set_ylabel('Digital Data (bits)', fontsize=10, fontweight = 'bold')
						axes1[2,1].grid()
						axes1[2,1].legend(loc='upper right', shadow=True,fontsize='10')
						axes1[2,1].set_ylim(0,2)
						axes1[2,1].text(0,1.0,"start1 is %1.1f"%start1,fontsize=10, fontweight = 'bold')
						axes1[2,1].text(0,1.1,"stop1 is %1.1f"%stop1,fontsize=10, fontweight = 'bold')
						axes1[2,1].text(0,1.2,"start2 is %1.1f"%start2,fontsize=10, fontweight = 'bold')
						axes1[2,1].text(0,1.3,"stop2 is %1.1f"%stop2,fontsize=10, fontweight = 'bold')
						axes1[2,1].text(0,1.4,"start3 is %1.1f"%start3,fontsize=10, fontweight = 'bold')
						axes1[2,1].text(0,1.5,"stop3 is %1.1f"%stop3,fontsize=10, fontweight = 'bold')
						avg_digital_data = np.mean(DigitalData1_in_a_shotProbe1,0)
						####################################PLOT OF DIGITAL DATA IN SHOT FROM RANDOM ATOM CYCLE###############
						x1 = np.arange(0,len(DigitalData1_in_a_shotProbe1[0,:]))
						axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[0,:]+.1,'-',color='navy',label="shot 1",linewidth=3.0)
						axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[75,:]+.2,'-',color='green',label="shot 76",linewidth=3.0)
						axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[150,:]+.3,'-',color='orange',label="shot 151",linewidth=3.0)
						# axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[225,:]+.4,'-',color='k',label="shot 226",linewidth=3.0)
						# axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[235,:]+.5,'-',color='red',label="shot 301",linewidth=3.0)
						# axes1[2,2].plot(x1,DigitalData1_in_a_shotProbe1[245,:]+.6,'-',color='blue',label="shot 375",linewidth=3.0)
						axes1[2,2].plot(x1,10*avg_digital_data+2,'s-',color='orange',label="averaged shot",linewidth=3.0)
						axes1[2,2].text(0,3,"Count1 is %i"%(digitalChannel1CounterProbe1),fontsize=10, fontweight = 'bold')
						axes1[2,2].set_title("Digital Data (1) in a shot for file 2", fontsize=10, fontweight='bold')
						axes1[2,2].set_xlabel('Index', fontsize=10, fontweight = 'bold')
						axes1[2,2].set_ylabel('Value (bit)', fontsize=10, fontweight = 'bold')
						axes1[2,2].text(0,1,"digital data delayed by %i shots"%delayshift)
						axes1[2,2].grid()
						axes1[2,2].set_ylim(0,5)
						plt.tight_layout()

				#Averages of things per file (100 atom cycles)
				OD_dir[i]=OD_file/SegmentsPerFile
				phase_shift_dir[i]=phase_shiftProbe1_File
				phase_shift_CLICK_dir[i]=phase_shift_CLICKProbe1_File 
				phase_shift_NOCLICK_dir[i]=phase_shift_NOCLICKProbe1_File
				amplitude_files+=amp_dir_test

				phase_shiftProbe1_File = 0
				phase_shift_CLICKProbe1_File = 0
				phase_shift_NOCLICKProbe1_File = 0
				i+=1


amplitude_files=amplitude_files/numFiles
final_amp=np.mean(amplitude_files,0)
# probePulsing_OD=-2*np.log(final_amp/mean_amplitude)
# np.savetxt(dir_main+"ODpulsing.csv",probePulsing_OD,delimiter = ',')
np.savetxt(dir_main+"amp_pulsing.csv",final_amp,delimiter = ',')

print("%i atom cycles analyzed" %numAtomCycles)
#print("%i phase noise" %np.mean(std_dir_test))
if Spectrum == True: 
	startspectrum = 1500
	stopspectrum = 4500
	offset_spectrum = -150+25
	phaseAVG = SegmentsPerFile*(phaseAVG)/numAtomCycles
	amplitudeAVG = SegmentsPerFile*amplitudeAVG/numAtomCycles
	amplitude_nomot = amplitudeAVG[startspectrum:stopspectrum]
	amplitude_mot = (amplitudeAVG[scansize+startspectrum-offset_spectrum:scansize+stopspectrum-offset_spectrum])
	phase_nomot = phaseAVG[startspectrum:stopspectrum] - np.mean(phaseAVG[startspectrum:stopspectrum])
	phase_mot = phaseAVG[scansize+startspectrum-offset_spectrum:scansize+stopspectrum-offset_spectrum] - np.mean(phaseAVG[scansize+startspectrum-offset_spectrum:scansize+stopspectrum-offset_spectrum])
	opticaldepth = -2*np.log(amplitude_mot/amplitude_nomot)-offsetOD
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
	axes1[1,1].text(5,0.5,"center is %1.3f" %center_spectrum) #added Dec 6, 2021 by Kyle
	axes1[1,1].text(5,0,"width is %1.1f" %width) #added Dec 6, 2021 by Kyle
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
axes1[2,2].text(0,4,"P1(N1) is %1.3f (%f)"%(P11,N1_dir),fontsize=10, fontweight='bold')

averaged_amplitude_in_a_shot_for_dir_final = averaged_amplitude_in_a_shot_for_dir/numAtomCycles
averaged_phase_in_a_shot_for_dir_final = averaged_phase_in_a_shot_for_dir/numAtomCycles
averaged_phase_in_a_shot_CLICK_for_dir_final = averaged_phase_in_a_shot_CLICK_for_dir/numAtomCycles
averaged_phase_in_a_shot_NOCLICK_for_dir_final = averaged_phase_in_a_shot_NOCLICK_for_dir/numAtomCycles

x3 = np.arange(0,len(averaged_phase_in_a_shot_for_dir_final))
xdata = np.append(np.arange(start1,stop1,1),np.arange(start3,stop3,1))


averaged_phase_in_a_shot_for_dir_final_zeromean = averaged_phase_in_a_shot_for_dir_final-np.mean(averaged_phase_in_a_shot_for_dir_final)
yPhaseData = np.append(averaged_phase_in_a_shot_for_dir_final_zeromean[start1:stop1],averaged_phase_in_a_shot_for_dir_final_zeromean[start3:stop3])
m,b = fit_to_a_line(xdata,yPhaseData)
averaged_phase_in_a_shot_for_dir_final_zeromean_noslope = averaged_phase_in_a_shot_for_dir_final_zeromean-x3*m-b-BackgroundPhase

averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean=averaged_phase_in_a_shot_CLICK_for_dir_final-np.mean(averaged_phase_in_a_shot_CLICK_for_dir_final)
yPhaseDataClick = np.append(averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean[start1:stop1],averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean[start3:stop3])
mClick,bClick = fit_to_a_line(xdata,yPhaseDataClick)
averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean_noslope = averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean-x3*mClick-bClick-BackgroundPhase

averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean=averaged_phase_in_a_shot_NOCLICK_for_dir_final-np.mean(averaged_phase_in_a_shot_NOCLICK_for_dir_final)
yPhaseDataNOClick = np.append(averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean[start1:stop1],averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean[start3:stop3])
mNOClick,bNOClick = fit_to_a_line(xdata,yPhaseDataNOClick)
averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean_noslope = averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean-x3*mNOClick-bNOClick-BackgroundPhase

average_amplitude = np.mean(averaged_amplitude_in_a_shot_for_dir_final)
amplitude_std = np.std(averaged_amplitude_in_a_shot_for_dir_final)
amplitudeshift = (np.mean(averaged_amplitude_in_a_shot_for_dir_final[start2:stop2]) - (np.mean(averaged_amplitude_in_a_shot_for_dir_final[start1:stop1])+np.mean(averaged_amplitude_in_a_shot_for_dir_final[start3:stop3]))/2.0)
PercentAmplitudeShift = 100*amplitudeshift/average_amplitude
if np.absolute(PercentAmplitudeShift) > 0.03:
	ExpectedCrossTalk = 0#PercentAmplitudeShift*259/100
else:
	ExpectedCrossTalk = 0

####################################PLOT OF AVG AMPLITUDE IN A SHOT ACROSS ALL ATOM CYCLES###############		
axes1[1,0].plot(x3,averaged_amplitude_in_a_shot_for_dir_final,'s-',color='navy',linewidth=3.0)
axes1[1,0].set_title("Amplitude avg in a shot for run", fontsize=10, fontweight='bold')
axes1[1,0].set_xlabel('Index', fontsize=10, fontweight = 'bold')
axes1[1,0].set_ylabel('Amplitude (mV)', fontsize=10, fontweight = 'bold')
axes1[1,0].text(0,  0.7*average_amplitude, r'Avg amp is %1.1f +/- %1.2f mrad' %(average_amplitude,amplitude_std), fontsize=12, fontweight='bold' )
axes1[1,0].text(0,  0.5*average_amplitude, r'Avg percent amp shift is %1.3f' %(PercentAmplitudeShift*factor), fontsize=12, fontweight='bold' )
axes1[1,0].text(0,  0.3*average_amplitude, r'Phase-amp cross talk is %1.2fmrad' %(ExpectedCrossTalk*factor), fontsize=12, fontweight='bold' )
axes1[1,0].grid()
axes1[1,0].set_ylim(0,1.1*max(averaged_amplitude_in_a_shot_for_dir_final))
axes1[1,0].axvline(x=start1,color='orange',linewidth=2.0)
axes1[1,0].axvline(x=stop1,color='orange',linewidth=2.0)
axes1[1,0].axvline(x=start2,color='orange',linewidth=2.0)
axes1[1,0].axvline(x=stop2,color='orange',linewidth=2.0)
axes1[1,0].axvline(x=start3,color='orange',linewidth=2.0)
axes1[1,0].axvline(x=stop3,color='orange',linewidth=2.0)
plt.tight_layout()


#Calculate STD of the mean using sums 
mean_XPS=phase_shiftProbe1_1/(numAtomCycles)
mean_square_XPS=square_phase_shift/(numAtomCycles)
phase_shift_dir_stem=1000*np.sqrt((mean_square_XPS-mean_XPS**2)/(numShots*numAtomCycles))

x2 = np.arange(0,len(phase_shift_dir))
phase_shift_dir_averaged = 1000*np.mean(phase_shift_dir)-ExpectedCrossTalk
#phase_shift_dir_stem = 1000*np.std(phase_shift_dir)/np.sqrt(len(phase_shift_dir)) #phase_shift_dir is the average XPS for each file analyzed (num elements = num files)
phase_shift_dir_std = 1000*np.std(phase_shift_dir)
phase_shift_CLICK_dir_averaged = 1000*np.mean(phase_shift_CLICK_dir)-ExpectedCrossTalk
phase_shift_CLICK_dir_stem = 1000*np.std(phase_shift_CLICK_dir)/np.sqrt(len(phase_shift_CLICK_dir))
phase_shift_CLICK_dir_std = 1000*np.std(phase_shift_CLICK_dir)
phase_shift_NOCLICK_dir_averaged = 1000*np.mean(phase_shift_NOCLICK_dir)-ExpectedCrossTalk
phase_shift_NOCLICK_dir_stem = 1000*np.std(phase_shift_NOCLICK_dir)/np.sqrt(len(phase_shift_NOCLICK_dir))
phase_shift_NOCLICK_dir_std = 1000*np.std(phase_shift_NOCLICK_dir)
CvNC = phase_shift_CLICK_dir_averaged*factor-phase_shift_NOCLICK_dir_averaged*factor
CvNC_std = np.sqrt((phase_shift_CLICK_dir_stem*factor)**2+(phase_shift_NOCLICK_dir_stem*factor)**2)



###################################PLOT OF AVG PHASE SHIFT PER SHOT FOR EVERY ATOM CYCLE###############		
axes1[0,2].plot(x2,1000*phase_shift_dir,'s',color='navy',linewidth=3.0)
axes1[0,2].set_title("Phase shift in a shot for the run", fontsize=10, fontweight='bold')
axes1[0,2].set_xlabel('shot #', fontsize=10, fontweight = 'bold')
axes1[0,2].set_ylabel('Shift (mrad)', fontsize=10, fontweight = 'bold')
axes1[0,2].grid()
axes1[0,2].text(1,  -15*phase_shift_dir_std, dir_main[0:40], fontsize=10, fontweight='bold' )
axes1[0,2].text(1,  -20*phase_shift_dir_std, dir_main[40:80], fontsize=10, fontweight='bold' )
axes1[0,2].text(1,  30*phase_shift_dir_std, r'averaged phase shift is %1.3f +/- %1.3f mrad' %((phase_shift_dir_averaged), (phase_shift_dir_stem)), fontsize=10, fontweight='bold' )
axes1[0,2].text(1,  4*phase_shift_dir_std, r'Uncompensated phase shift is %1.3f +/- %1.3f (%1.2f) mrad' %(phase_shift_dir_averaged, phase_shift_dir_stem,phase_shift_dir_std), fontsize=10, fontweight='bold' )
axes1[0,2].text(1,  12*phase_shift_dir_std, r'phase shift for CLICK is %1.3f +/- %1.3f (%1.2f) mrad' %(phase_shift_CLICK_dir_averaged*factor, phase_shift_CLICK_dir_stem*factor,phase_shift_CLICK_dir_std*factor), fontsize=10, fontweight='bold' )
axes1[0,2].text(1,  8*phase_shift_dir_std, r'phase shift for NO CLICK is %1.3f +/- %1.3f (%1.2f) mrad' %(phase_shift_NOCLICK_dir_averaged*factor, phase_shift_NOCLICK_dir_stem*factor,phase_shift_NOCLICK_dir_std*factor), fontsize=10, fontweight='bold' )
axes1[0,2].text(1,  16*phase_shift_dir_std, r'phase shift  is %1.3f +/- %1.3f (%1.2f) mrad' %(phase_shift_dir_averaged*factor, phase_shift_dir_stem*factor,phase_shift_dir_std*factor), fontsize=10, fontweight='bold' )
axes1[0,2].text(1,  20*phase_shift_dir_std, r'Click vs No click is %1.3f +/- %1.3f mrad' %(CvNC,CvNC_std), fontsize=10, fontweight='bold' )
axes1[0,2].text(1,  24*phase_shift_dir_std, r'%i measurements' %(len(phase_shift_dir)), fontsize=10, fontweight='bold' )
axes1[0,2].set_ylim(-30*phase_shift_dir_std,45*phase_shift_dir_std)
#plt.tight_layout()	



axes11ylim = 1.5*max(abs(5*np.min(averaged_phase_in_a_shot_for_dir_final_zeromean)),5*np.max(averaged_phase_in_a_shot_for_dir_final_zeromean),2e-4)
offset1=axes11ylim/3
phaseshift_otherway = 1000*(np.mean(averaged_phase_in_a_shot_for_dir_final_zeromean_noslope[start2:stop2]))
measured = phaseshift_otherway*factor-ExpectedCrossTalk*factor


CvNC_difference = averaged_phase_in_a_shot_CLICK_for_dir_final_zeromean_noslope-averaged_phase_in_a_shot_NOCLICK_for_dir_final_zeromean_noslope

###################################PLOT OF AVG PHASE IN A SHOT ACROSS ALL ATOM CYCLES###############		
axes1[1,2].plot(x3,averaged_phase_in_a_shot_for_dir_final_zeromean,'s-',color='orange',linewidth=1.0)
axes1[1,2].plot(x3,averaged_phase_in_a_shot_for_dir_final_zeromean_noslope,'s-',color='navy',linewidth=3.0)
axes1[1,2].plot(x3,CvNC_difference+offset1,'s-',color='navy',linewidth=3.0)
axes1[1,2].set_title("Phase avg in a shot for run", fontsize=10, fontweight='bold')
axes1[1,2].set_xlabel('Index', fontsize=10, fontweight = 'bold')
axes1[1,2].set_ylabel('Phase (rad)', fontsize=10, fontweight = 'bold')
axes1[1,2].text(1,  -.3*axes11ylim, r'Phase shifts are %1.3f +/- %1.5f mrad' %(phaseshift_otherway*factor,phase_shift_dir_stem*factor), fontsize=10, fontweight='bold' )
axes1[1,2].text(1,  -.5*axes11ylim, r'Phase shifts are corrected %1.3f +/- %1.3f mrad' %(measured,phase_shift_dir_stem*factor), fontsize=10, fontweight='bold' )
axes1[1,2].text(1,  -.8*axes11ylim, r'std in shot is %1.3furad' %(1e6*np.std(averaged_phase_in_a_shot_for_dir_final_zeromean_noslope)), fontsize=10, fontweight='bold' )
axes1[1,2].axvline(x=start1,color='orange',linewidth=2.0)
axes1[1,2].axvline(x=stop1,color='orange',linewidth=2.0)
axes1[1,2].axvline(x=start2,color='orange',linewidth=2.0)
axes1[1,2].axvline(x=stop2,color='orange',linewidth=2.0)
axes1[1,2].axvline(x=start3,color='orange',linewidth=2.0)
axes1[1,2].axvline(x=stop3,color='orange',linewidth=2.0)
axes1[1,2].grid()
axes1[1,2].set_ylim(-axes11ylim,axes11ylim)
#plt.tight_layout()
stringname = dir_main+".csv"
stringname2 = stringname.replace('/','_')
stringname2 = stringname2[8:]
stringname22 = dir_main+"XPS.csv"
stringname23 = dir_main+"CvNC.csv"
stringname24 = dir_main+"SPCM.csv"
stringname25 = dir_main+"Cphase.csv"
stringname26 = dir_main+"NCphase.csv"

np.savetxt(stringname23,(CvNC_difference),delimiter = ',')
np.savetxt(stringname22,(averaged_phase_in_a_shot_for_dir_final_zeromean_noslope),delimiter=',')
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
#print(m,b)
# m=-0.0003972738128928087
# b=-1455.6695568959215
avg_phase_in_a_cycle=avg_phase_in_a_cycle-(m*ch1_x[2*scansize+72:numMeasurementsTotal]+b)
stringnamepng = dir_main+"phase_in_a_cycle.csv"
np.savetxt(stringnamepng,(avg_phase_in_a_cycle),delimiter = ',')

plt.figure(figsize=(7.2,5))
#plt.axhline(y=0, color='b', linestyle='-')
plt.xlabel(r"measurement", fontsize = 18)
plt.ylabel(r'phase', fontsize = 18)
plt.plot(np.arange(len(avg_phase_in_a_cycle)),avg_phase_in_a_cycle,'go')
stringnamepng = dir_main+"average_phase.png"
plt.savefig(stringnamepng,format='png', dpi=400)
plt.show()

Phase_in_a_shot = avg_phase_in_a_cycle.reshape(numShots,numMeasurementsPerShot)

#subtract off linear background
background_i_phase = np.mean(Phase_in_a_shot[:,start1:stop1],1)
background_f_phase = np.mean(Phase_in_a_shot[:,start3:stop3],1)
background_phase =  (background_i_phase+background_f_phase)/2
signal_phase = np.mean(Phase_in_a_shot[:,start2:stop2],1) #getting peak value
phase_shift = 1000*(signal_phase - background_phase)
average_shots=103
phase_shift_shots=np.mean(np.reshape(phase_shift,(int(numShots/average_shots),average_shots)),1)
shots_std=np.std(np.reshape(phase_shift,(int(numShots/average_shots),average_shots)),1)
# stringnamepng = dir_main+"XPS_shots_array.csv"
# np.savetxt(stringnamepng,(phase_shift),delimiter = ',')
# stringnamepng = dir_main+"XPS_shots_average.csv"
# np.savetxt(stringnamepng,(phase_shift_shots,shots_std),delimiter = ',')

phase_in_shot_over_time=Phase_in_a_shot.reshape(-1,average_shots,numMeasurementsPerShot).mean(axis=1)

# for i in range(int(numShots/average_shots)):
# 	plt.figure()
# 	plt.grid()
# 	plt.ylim(-8,8)
# 	plt.plot(np.arange(numMeasurementsPerShot),1000*(phase_in_shot_over_time[i]-phase_in_shot_over_time[i].mean()))
# 	stringnamepng = dir_main+"XPS_cycle"+str(i)+".png"
# 	plt.savefig(stringnamepng,format='png', dpi=400)


print(dir_main)
stoptime = time.time()
print("Program took %1.2f seconds" %(stoptime-starttime))

# plt.figure(figsize=(7.2,5))
# plt.axhline(y=0, color='b', linestyle='-')
# plt.xlabel(r"shot", fontsize = 18)
# plt.ylabel(r'XPS', fontsize = 18)
# #plt.text(700,phase_shift.max(),r"new mean is %1.2f" %(np.mean(phase_shift[0:800])))
# plt.errorbar(np.arange(numShots/average_shots),phase_shift_shots,yerr=shots_std/np.sqrt(average_shots),fmt='ro')
# #plt.legend((r'Signal off',r'Signal on'))
# stringnamepng = dir_main+"XPS_cycle.png"
# plt.savefig(stringnamepng,format='png', dpi=400)
# #plt.show()

# #this is temporary code to find the click rates or OD for each shot as a function of time (signal info)
# stringnamepng = dir_main+"clicks_cycle.csv"
# clicks_pershot=np.sum(DigitalData1_cycles,1)/numAtomCycles
# log_clicks_pershot=-np.log(clicks_pershot)
# log_grouped=np.mean(np.reshape(log_clicks_pershot,(int(numShots/average_shots),average_shots)),1)
# clicks_grouped_std=np.std(np.reshape(log_clicks_pershot,(int(numShots/average_shots),average_shots)),1)
# np.savetxt(stringnamepng,(clicks_pershot),delimiter = ',')
# plt.figure(figsize=(7.2,5))
# plt.errorbar(np.arange(numShots/average_shots),log_grouped,yerr=clicks_grouped_std/np.sqrt(average_shots),fmt='ro')
# stringnamepng = dir_main+"clicks_cycle.png"
# plt.savefig(stringnamepng,format='png', dpi=400)
# plt.show()