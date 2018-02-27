import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import  sequence

import tensorflow as tf
session_conf = tf.ConfigProto(
	intra_op_parallelism_threads=1,
	inter_op_parallelism_threads=1
)

np.random.seed(42)
tf.set_random_seed(42)

FILE_PATH_ACC = '/Users/adriagil/UNI/IPAL/MobilityApps/HAR/Activitrack_DNN/UCI HAR Dataset2/Phones_accelerometer.csv'
FILE_PATH_GYR = '/Users/adriagil/UNI/IPAL/MobilityApps/HAR/Activitrack_DNN/UCI HAR Dataset2/Phones_gyroscope.csv'
columns2Normalize = ['x', 'y', 'z']

# defining function for loading the dataset
def readData(filePath):
	# attributes of the dataset
	data = pd.read_csv(filePath, dtype={'Index': str, 'Arrival_Time': np.float64, 'Creation_Time': np.float64, 'x': np.float64, 'y': np.float64,'z': np.float64, 
		'User': str, 'Model': str, 'Device': str, 'gt': str})
	return data

# defining a function for feature normalization --> (feature - mean)/stdiv
def featureNormalize(dataset):
	mu = np.mean(dataset, axis=0)
	sigma = np.std(dataset, axis=0)
	return (dataset-mu)/sigma


print('Loading data ...')
data1 = readData(FILE_PATH_ACC)		#Loading Accelerometer data
data2 = readData(FILE_PATH_GYR)			#Loading Gyroscope data
print("Data Loaded Succesfully!")

length1 = len(data1)
length2 = len(data2)

print("Phones_accelerometer length: " + str(length1) + "\nPhones_gyroscope length: " + str(length2))

data1[x] = [featureNormalize(data1[x]) for x in columns2Normalize]
data2[x] = [featureNormalize(data2[x]) for x in columns2Normalize]
print("Data Normalized")




