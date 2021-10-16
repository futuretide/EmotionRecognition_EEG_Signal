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

	def send_result_to_application(self,emotion_class):
		"""
		Send emotion predict to web app.
		Input: Class of emotion between 1 to 5 according to Russel's Circumplex Model.
		Output: Send emotion prediction to web app.
		"""
		socket =  SocketIO('localhost', socket_port, LoggingNamespace)
		socket.emit('realtime emotion',emotion_class)

	def main_process(self):
		"""
		Get realtime EEG data from Emotiv EPOC, process all data (FFT, feature extraction, and classification), and predict the emotion.
		Input: -
		Output: Class of emotion between 1 to 5 according to Russel's Circumplex Model.
		"""
		headset = Emotiv()
		gevent.spawn(headset.setup)
		gevent.sleep(0)

		threads = []
		eeg_realtime = np.zeros((number_of_channel,number_of_realtime_eeg),dtype=np.double)
		counter=0
		init=True

		try:
			#Looping to get realtime EEG data from Emotiv EPOC
		    while True:
		        packet = headset.dequeue()

		        #Get initial EEG data for all channels
		        if init:
		        	for i in range(number_of_channel):eeg_realtime[i,counter]=packet.sensors[channel_names[i]]['value']
		        else:
		        	new_data=[packet.sensors[channel_names[i]]['value'] for i in range(number_of_channel)]
		        	eeg_realtime=np.insert(eeg_realtime,number_of_realtime_eeg,new_data,axis=1)
		        	eeg_realtime=np.delete(eeg_realtime,0,axis=1)
		        
		        #If EEG data have been recorded in ... seconds, then process data to predict emotion
		        if counter == (sampling_rate-1) or counter == (number_of_realtime_eeg-1):
		        	t = threading.Thread(target=rte.process_all_data, args=(eeg_realtime,))
		        	threads.append(t)
		        	t.start()

		        	init=False
		        	counter=0

		        gevent.sleep(0)
		        counter += 1

		except KeyboardInterrupt:
		    headset.close()
		finally:
		    headset.close()



if _name_ == "_main_":
	rte = RealtimeEmotion()
	rte.main_process()
