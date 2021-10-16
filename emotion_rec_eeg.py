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
	def get_feature(self,all_channel_data):
		"""
		Get feature from each frequency.
		Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
		Output: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
		"""
		#Get frequency data
		(delta,theta,alpha,beta,gamma) = self.get_frequency(all_channel_data)

		#Compute feature std
		delta_std = np.std(delta, axis=1)
		theta_std = np.std(theta, axis=1)
		alpha_std = np.std(alpha, axis=1)
		beta_std = np.std(beta, axis=1)
		gamma_std = np.std(gamma, axis=1)

		#Compute feature means
		delta_m = np.mean(delta, axis=1)
		theta_m = np.mean(theta, axis=1)
		alpha_m = np.mean(alpha, axis=1)
		beta_m = np.mean(beta, axis=1)
		gamma_m = np.mean(gamma, axis=1)

		#Concate feature
		feature = np.array([delta_std,delta_m,theta_std,theta_m,alpha_std,alpha_m,beta_std,beta_m,gamma_std,gamma_m])
		feature = feature.T
		feature = feature.ravel()

	return feature

	def predict_emotion(self,feature):
		"""
		Get arousal and valence class from feature.
		Input: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
		Output: Class of emotion between 1 to 3 from each arousal and valence. 1 denotes low category, 2 denotes normal category, and 3 denotes high category.
		"""
		#Compute canberra with arousal training data
		distance_ar = map(lambda x:ss.distance.canberra(x,feature),self.train_arousal)

		#Compute canberra with valence training data
		distance_va = map(lambda x:ss.distance.canberra(x,feature),self.train_valence)

		#Compute 3 nearest index and distance value from arousal
		idx_nearest_ar = np.array(np.argsort(distance_ar)[:3])
		val_nearest_ar = np.array(np.sort(distance_ar)[:3])

		#Compute 3 nearest index and distance value from arousal
		idx_nearest_va = np.array(np.argsort(distance_va)[:3])
		val_nearest_va = np.array(np.sort(distance_va)[:3])

		#Compute comparation from first nearest and second nearest distance. If comparation less or equal than 0.7, then take class from the first nearest distance. Else take frequently class.
		#Arousal
		comp_ar = val_nearest_ar[0]/val_nearest_ar[1]
		if comp_ar<=0.97:
			result_ar = self.class_arousal[0,idx_nearest_ar[0]]
		else:
			result_ar = sst.mode(self.class_arousal[0,idx_nearest_ar])
			result_ar = float(result_ar[0])

		#Valence
		comp_va = val_nearest_va[0]/val_nearest_va[1]
		if comp_va<=0.97:
			result_va = self.class_valence[0,idx_nearest_va[0]]
		else:
			result_va = sst.mode(self.class_valence[0,idx_nearest_va])
			result_va = float(result_va[0])

		return result_ar,result_va


if _name_ == "_main_":
	rte = RealtimeEmotion()
	rte.main_process()
