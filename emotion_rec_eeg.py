import csv
import numpy as np
import scipy.spatial as ss
import scipy.stats as sst
from emokit.emotiv import Emotiv
import platform
import socket
import gevent
import threading
from socketIO_client import SocketIO, LoggingNamespace

sampling_rate = 128  #In hertz
number_of_channel = 14
realtime_eeg_in_second = 5 #Realtime each ... seconds
number_of_realtime_eeg = sampling_rate*realtime_eeg_in_second
socket_port = 8080

channel_names=[
	"AF3",
	"F7",
	"F3",
	"FC5",
	"T7",
	"P7",
	"O1",
	"O2",
	"P8",
	"T8",
	"FC6",
	"F4",
	"F8",
	"AF4"
]

class RealtimeEmotion(object): 
	"""
	Receives EEG data realtime, preprocessing and predict emotion.
	"""

	# path is set to training data directory
	def _init_(self, path="../Training Data/"): 
		"""
		Initializes training data and their classes.
		"""
		self.train_arousal = self.get_csv(path + "train_arousal.csv")
		self.train_valence = self.get_csv(path + "train_valence.csv")
		self.class_arousal = self.get_csv(path + "class_arousal.csv")
		self.class_valence = self.get_csv(path + "class_valence.csv")

	def get_csv(self,path): 
		"""
		Get data from csv and convert them to numpy python.
		Input: Path csv file.
		Output: Numpy array from csv data.
		"""
		#Get csv data to list
		file_csv = open(path)
		data_csv = csv.reader(file_csv)
		data_training = np.array([each_line for each_line in data_csv])

		#Convert list to float
		data_training = data_training.astype(np.double)

		return data_training

	def do_fft(self,all_channel_data): 
		"""
		Do fft in each channel for all channels.
		Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
		Output: FFT result with dimension N x M. N denotes number of channel and M denotes number of FFT data from each channel.
		"""
		data_fft = map(lambda x: np.fft.fft(x),all_channel_data)

		return data_fft

	def get_frequency(self,all_channel_data): 
		"""
		Get frequency from computed fft for all channels. 
		Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
		Output: Frequency band from each channel: Delta, Theta, Alpha, Beta, and Gamma.
		"""
		#Length data channel
		L = len(all_channel_data[0])

		#Sampling frequency
		Fs = 128

		#Get fft data
		data_fft = self.do_fft(all_channel_data)

		#Compute frequency
		frequency = map(lambda x: abs(x/L),data_fft)
		frequency = map(lambda x: x[: L/2+1]*2,frequency)

		#List frequency
		delta = map(lambda x: x[L*1/Fs-1: L*4/Fs],frequency)
		theta = map(lambda x: x[L*4/Fs-1: L*8/Fs],frequency)
		alpha = map(lambda x: x[L*5/Fs-1: L*13/Fs],frequency)
		beta = map(lambda x: x[L*13/Fs-1: L*30/Fs],frequency)
		gamma = map(lambda x: x[L*30/Fs-1: L*50/Fs],frequency)

		return delta,theta,alpha,beta,gamma
	
	def determine_emotion_class(self,feature):
		"""
		Get emotion class from feature.
		Input: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
		Output: Class of emotion between 1 to 5 according to Russel's Circumplex Model.
		"""
		class_ar,class_va = self.predict_emotion(feature)

		if class_ar==2.0 or class_va==2.0:
			emotion_class = 5
		elif class_ar==3.0 and class_va==1.0:
			emotion_class = 1
		elif class_ar==3.0 and class_va==3.0:
			emotion_class = 2
		elif class_ar==1.0 and class_va==3.0:
			emotion_class = 3
		elif class_ar==1.0 and class_va==1.0:
			emotion_class = 4

		return emotion_class

	def process_all_data(self,all_channel_data):
		"""
		Process all data from EEG data to predict emotion class.
		Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
		Output: Class of emotion between 1 to 5 according to Russel's Circumplex Model. And send it to web ap
		"""
		#Get feature from EEG data
		feature = self.get_feature(all_channel_data)

		#Predict emotion class
		emotion_class = self.determine_emotion_class(feature)

		#send emotion_class to web app
		self.send_result_to_application(emotion_class)



if _name_ == "_main_":
	rte = RealtimeEmotion()
	rte.main_process()
