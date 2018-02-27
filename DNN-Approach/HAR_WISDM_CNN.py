#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# importing libraries and dependecies 
import pandas as pd
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
session_conf = tf.ConfigProto(
	intra_op_parallelism_threads=1,
	inter_op_parallelism_threads=1
)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
#from keras import backend as K
from keras import optimizers
#K.set_image_dim_ordering('th')
# setting up a random seed for reproducibility
random_seed = 611
np.random.seed(random_seed)

# defining function for loading the dataset
def readData(filePath):
	# attributes of the dataset
	columnNames = ['user_id', 'activity', 'timestamp', 'x-axis', 'y-axis','z-axis']
	print("Reading csv...")
	data = pd.read_csv(filePath, header = None, names=columnNames, lineterminator=';', 
		dtype={'user_id': str, 'activity': str, 'timestamp': np.float64, 'x-axis': np.float64, 'y-axis': np.float64,'z-axis': np.float64})
	print("Data Loaded Succesfully!")
	return data

# defining a function for feature normalization
# (feature - mean)/stdiv
def featureNormalize(dataset):
	mu = np.mean(dataset, axis=0)
	sigma = np.std(dataset, axis=0)
	return (dataset-mu)/sigma

# defining a window function for segmentation purposes
def windows(data,size):
	start = 0
	while start< data.count():
		yield int(start), int(start + size)
		start += (size/2)

# segmenting the time series
def segment_signal(data, window_size = 90):
	segments = np.empty((0,window_size,3))
	labels= np.empty((0))
	count = 0
	for (start, end) in windows(data['timestamp'], window_size):
		x = data['x-axis'][start:end]
		y = data['y-axis'][start:end]
		z = data['z-axis'][start:end]
		count += 1
		if(len(data['timestamp'][start:end]) == window_size):
			segments = np.vstack([segments,np.dstack([x,y,z])])
			labels = np.append(labels,stats.mode(data['activity'][start:end])[0][0])
		if (count % 5000) == 0:
			print("Segmenting...")
	return segments, labels

def cnnModel():
	
	return model

def main():
	''' Main Code '''
	# # # # # # # # #   reading the data   # # # # # # # # # # 
	# Path of file #
	dataset = readData('WISDM_at_v2.0/WISDM_at_v2.0_raw.txt')
	#dataset = readData('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
	dataset['x-axis'] = featureNormalize(dataset['x-axis'])
	dataset['y-axis'] = featureNormalize(dataset['y-axis'])
	dataset['z-axis'] = featureNormalize(dataset['z-axis'])
	print("Data Normalized")

	# segmenting the signal in overlapping windows of 90 samples with 50% overlap
	segments, labels = segment_signal(dataset) 
	print("Data Segmented")

	with open('dataSegmentedWISDM_raw  .pkl', 'wb') as f:
		pickle.dump([segments, labels], f)

	#with open('dataSegmentedWISDM.pkl', 'rb') as f:
	#	segments, labels = pickle.load(f)

	#categorically defining the classes of the activities
	labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
	# defining parameters for the input and network layers
	# we are treating each segmeent or chunk as a 2D image (90 X 3)
	numOfRows = segments.shape[1]
	numOfColumns = segments.shape[2]
	numChannels = 1
	numFilters = 128 # number of filters in Conv2D layer
	# kernal size of the Conv2D layer
	kernalSize1 = 2
	# max pooling window size
	poolingWindowSz = 2
	# number of filters in fully connected layers
	numNueronsFCL1 = 128
	numNueronsFCL2 = 128
	# split ratio for test and validation
	trainSplitRatio = 0.8
	# number of epochs
	Epochs = 10
	# batchsize
	batchSize = 32
	# number of total clases
	numClasses = labels.shape[1]
	# dropout ratio for dropout layer
	dropOutRatio = 0.5
	# reshaping the data for network input
	reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns,1)
	# splitting in training and testing data
	trainSplit = np.random.rand(len(reshapedSegments)) < trainSplitRatio
	trainX = reshapedSegments[trainSplit]
	testX = reshapedSegments[~trainSplit]
	trainX = np.nan_to_num(trainX)
	testX = np.nan_to_num(testX)
	trainY = labels[trainSplit]
	testY = labels[~trainSplit]

	#CNN Model
	model = Sequential()
	# adding the first convolutionial layer with 32 filters and 5 by 5 kernal size, using the rectifier as the activation function
	model.add(Conv2D(numFilters, (kernalSize1,kernalSize1),input_shape=(numOfRows, numOfColumns,1),activation='relu'))
	# adding a maxpooling layer
	model.add(MaxPooling2D(pool_size=(poolingWindowSz,poolingWindowSz),padding='valid'))
	# adding a dropout layer for the regularization and avoiding over fitting
	model.add(Dropout(dropOutRatio))
	# flattening the output in order to apply the fully connected layer
	model.add(Flatten())
	# adding first fully connected layer with 256 outputs
	model.add(Dense(numNueronsFCL1, activation='relu'))
	#adding second fully connected layer 128 outputs
	model.add(Dense(numNueronsFCL2, activation='relu'))
	# adding softmax layer for the classification
	model.add(Dense(numClasses, activation='softmax'))
	# Compiling the model to generate a model
	adam = optimizers.Adam(lr = 0.001, decay=1e-6)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

	for layer in model.layers:
		print(layer.name)

	model.fit(trainX,trainY, validation_split=1-trainSplitRatio,epochs=10,batch_size=batchSize)
	score = model.evaluate(testX,testY,verbose=0)

	print('Baseline Error: %.2f%%' %(100-score[1]*100))
	model.save('model.h5')
	np.save('groundTruth.npy',testY)
	np.save('testData.npy',testX)


if __name__ == '__main__':
	main()

