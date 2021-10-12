"""==================================================================
================Main: =====================
=====================================================================

Started: Mar 30, 2021

Written by: Kyle Thompson

Program Name: 

Description: This is going to be a class for storing information contained in a TDMS file.

Completed: 

Last Updated: April 1, 2021 (added range_number function)

=====================================================================
=====================================================================
"""

import numpy as np
from nptdms import TdmsFile

#function to get voltage range from string
def range_number(range_word):
	switcher={
		"±200 mV":200,
		"±500 mV":500,
		"±1000 mV":1000,
		"±2000 mV":2000,
		"±5000 mV":5000,
		"±10000 mV":10000
	} #dictionary to convert strings to numbers
	return switcher.get(range_word, "Error: Invalid range")


class Spectrum_info_class:
	def __init__(self, filename):
		tdms_file = TdmsFile(filename) #open the file
		self.Mem = tdms_file.object('Raw Data').properties["Mem"]
		self.Seg = tdms_file.object('Raw Data').properties["Seg"]
		self.Pre = tdms_file.object('Raw Data').properties["Pre"]
		self.Post = tdms_file.object('Raw Data').properties["Post"]
		self.num_segments = int(self.Mem/self.Seg)
		self.sampling_rate = tdms_file.object('Raw Data').properties["Sampling Rate (kHz)"]
		self.trigger_source = tdms_file.object('Raw Data').properties["Trigger Source"]
		#channel info
		self.ch0_voltage_range = tdms_file.object('Raw Data','Channel 0').properties["Input Range "] #the space is needed here
		self.ch1_voltage_range = tdms_file.object('Raw Data','Channel 1').properties["Input Range "] #the space is needed here
		self.ch2_voltage_range = tdms_file.object('Raw Data','Channel 2').properties["Input Range "] #the space is needed here
		self.ch3_voltage_range = tdms_file.object('Raw Data','Channel 3').properties["Input Range "] #the space is needed here
		self.ch0_voltage_range_number = range_number(self.ch0_voltage_range)  #get number, instead of word
		self.ch1_voltage_range_number = range_number(self.ch1_voltage_range) 
		self.ch2_voltage_range_number = range_number(self.ch2_voltage_range) 
		self.ch3_voltage_range_number = range_number(self.ch3_voltage_range) 
		self.ch0_termination = tdms_file.object('Raw Data','Channel 0').properties["Termination"]
		self.ch1_termination = tdms_file.object('Raw Data','Channel 1').properties["Termination"]
		self.ch2_termination = tdms_file.object('Raw Data','Channel 2').properties["Termination"]
		self.ch3_termination = tdms_file.object('Raw Data','Channel 3').properties["Termination"]
		self.ch0_enabled = tdms_file.object('Raw Data','Channel 0').properties["Enable"]
		self.ch1_enabled = tdms_file.object('Raw Data','Channel 1').properties["Enable"]
		self.ch2_enabled = tdms_file.object('Raw Data','Channel 2').properties["Enable"]
		self.ch3_enabled = tdms_file.object('Raw Data','Channel 3').properties["Enable"]
		self.num_channels_enabled = self.ch0_enabled+self.ch1_enabled+self.ch2_enabled+self.ch3_enabled
		#multipurpose channels
		self.X0_mode = tdms_file.object('Raw Data').properties["X0 Mode"]
		self.X1_mode = tdms_file.object('Raw Data').properties["X1 Mode"]
		self.X2_mode = tdms_file.object('Raw Data').properties["X2 Mode"]
		self.X3_mode = tdms_file.object('Raw Data').properties["X3 Mode"]	
		self.digital_data = (self.X1_mode == "Digital In") #whether or not digital data was recorded

	#define function to print everything	
	def print_info(self):
		print("Mem = {} samples".format(self.Mem))
		print("Seg = {} samples".format(self.Seg))
		print("Seg = {} samples".format(self.Seg))
		print("Pre = {} samples".format(self.Pre))
		print("Post = {} samples".format(self.Post))
		print("Sampling rate (kHz) = {}".format(self.sampling_rate))
		print("Trigger source = {}".format(self.trigger_source))
		print("Number of segments = {}".format(self.num_segments))

		print("ch0 range = " + self.ch0_voltage_range)
		print("ch1 range = " + self.ch1_voltage_range)
		print("ch2 range = " + self.ch2_voltage_range)
		print("ch3 range = " + self.ch3_voltage_range)
		print("ch0 termination = {}".format(self.ch0_termination))
		print("ch1 termination = {}".format(self.ch1_termination))
		print("ch2 termination = {}".format(self.ch2_termination))
		print("ch3 termination = {}".format(self.ch3_termination))
		print("Is ch0 enabled? (0 for no, 1 for yes): {}".format(self.ch0_enabled))
		print("Is ch1 enabled? (0 for no, 1 for yes): {}".format(self.ch1_enabled))
		print("Is ch2 enabled? (0 for no, 1 for yes): {}".format(self.ch2_enabled))
		print("Is ch3 enabled? (0 for no, 1 for yes): {}".format(self.ch3_enabled))
		print("Number of enabled channels = {}".format(self.num_channels_enabled))

		print("X0_mode = {}".format(self.X0_mode))
		print("X1_mode = {}".format(self.X1_mode))
		print("X2_mode = {}".format(self.X2_mode))
		print("X3_mode = {}".format(self.X3_mode))
		print("Digital data? = {}".format(self.digital_data))
