import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
import pandas as pd

session_conf = tf.ConfigProto(
	intra_op_parallelism_threads=1,
	inter_op_parallelism_threads=1
)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout

import os

# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
#	"body_acc_x_",
#	"body_acc_y_",
#	"body_acc_z_",
	"body_gyro_x_",
	"body_gyro_y_",
	"body_gyro_z_",
	"total_acc_x_",
	"total_acc_y_",
	"total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
	"WALKING", 
	"WALKING_UPSTAIRS", 
	"WALKING_DOWNSTAIRS", 
	"SITTING", 
	"STANDING", 
	"LAYING"
] 

DATA_PATH = "/Users/adriagil/UNI/IPAL/MobilityApps/HAR/"
DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
print("\n" + "Dataset is now located at: " + DATASET_PATH)

TRAIN = "train/"
TEST = "test/"

def _read_csv(filename):
	return pd.read_csv(filename, delim_whitespace=True, header=None)

# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
	X_signals = []
	
	for signal_type_path in X_signals_paths:
		file = open(signal_type_path, 'r')
		# Read dataset from disk, dealing with text files' syntax
		X_signals.append(
			[np.array(serie, dtype=np.float32) for serie in [
				row.replace('  ', ' ').strip().split(' ') for row in file
			]]
		)
		file.close()
	
	return np.transpose(np.array(X_signals), (1, 2, 0))


# Load "y" (the neural network's training and testing outputs)
def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y = _read_csv(file)[0]

    print(y)

    return pd.get_dummies(y).as_matrix()


def main():

	X_train_signals_paths = [
	DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
	]
	X_test_signals_paths = [
		DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
	]

	X_train = load_X(X_train_signals_paths)
	X_test = load_X(X_test_signals_paths)

	y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
	y_test_path = DATASET_PATH + TEST + "y_test.txt"

	Y_train = load_y(y_train_path)
	Y_test = load_y(y_test_path)

	print(Y_test)

	# Input Data 
	training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
	test_data_count = len(X_test)  # 2947 testing series
	n_steps = len(X_train[0])  # 128 timesteps per series
	n_input = len(X_train[0][0])  # 6 input parameters per timestep

	# LSTM Neural Network's internal structure
	n_hidden = 32 # Hidden layer num of features
	n_classes = 6 # Total classes (should go up, or should go down)

	# Training 
	learning_rate = 0.0025
	lambda_loss_amount = 0.0015
	training_iters = training_data_count * 300  # Loop 300 times on the dataset
	epochs = 100
	batch_size = 128
	display_iter = 30000  # To show test set accuracy during training

	AccuracyArr = np.empty(0)
	sArr = np.empty(0)

	model = Sequential()
	model.add(LSTM(n_hidden, input_shape=(n_steps, n_input)))
	model.add(Dropout(0.2))
	model.add(Dense(n_classes, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	print(model.summary())
	model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))

	# Final evaluation of the model
	scores = model.evaluate(X_test, Y_test, verbose=0)
	AccuracyArr = np.append(AccuracyArr,(scores[1]*100))
	print("Accuracy: %.2f%%" % (scores[1]*100))


if __name__ == '__main__':
	main()