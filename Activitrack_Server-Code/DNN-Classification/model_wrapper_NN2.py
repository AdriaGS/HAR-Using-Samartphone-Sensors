import pandas as pd
import pickle
import optparse
import preprocessing as pp
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import tree
from keras.models import load_model
from sklearn import preprocessing

class ModelWrapper:
	'''
	Manage and use an activity classifier.
	'''

	_DEFAULT_MODEL_NAME = 'default.pck'
	_FEATURES_TO_USE = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z']


	def __init__(self):
		self.__loaded_model = None
		pass


	def load_default_model(self):
		print("Not implemented")
		return None


	def load_model(self, path):
		'''
		Load the model stored in the path passed.
		This method needs to be called before trying to predict or train.
		'''
		self.__loaded_model = load_model(path)
		return self.__loaded_model


	def _check_model_loaded(self):
		'''
		Check if there is a model loaded and load the default one if there isn't one.
		'''
		if self.__loaded_model == None:
			print("No model loaded! Call load_model(path) first.")
			# print("No model loaded! Using the default one")
			self.__loaded_model = load_default_model()

	def get_scaled(self, df):
		x = df.values
		x_scaled = preprocessing.StandardScaler().fit_transform(x)
		df_s = pd.DataFrame(x_scaled)
		return df_s

	def predict_json(self, rows):
		'''
		Predict the activity label(s) for the row(s) of the passed json.
		'''
		self._check_model_loaded()
		#rows_df = self.processed_df_from_dict(rows)
		df = pd.read_json(rows)
		#print(df.head())
		feat, clf = self.__loaded_model.values()
		df = pp.post_process_df(df)[feat]
		res = clf.predict(df)
		return res


	def predict_csv(self, csv_file):
		'''
		Predict the activity label(s) for the row(s) of the passed CSV file.
		'''
		self._check_model_loaded()
		df = pd.read_csv(csv_file, na_values='null')
		df = df.fillna(0)
		df = df.drop(df[(df["android.sensor.gravity"] == 0) | (df["android.sensor.accelerometer"] == 0)].index)
		#print(df)
		df = pp.post_process_df(df)
		
		ts = df.timestamp
		X_s = self.get_scaled(df[self._FEATURES_TO_USE])
		n_steps = 128
		n_input = 9

		X_s = pd.get_dummies(X_s)
		X_s = np.asarray(X_s)
		
		if len(X_s) > n_steps:
			X = X_s[int(len(X_s)/2) - int(n_steps/2):int(len(X_s)/2) + int(n_steps/2), :]
		
		X = X_s.reshape(X_s.shape[0],1,X_s.shape[1])

		prediction =  self.__loaded_model.predict(X)
		print(prediction)
		#print(zip(ts, prediction)[:5])
		return prediction


def main():
	print("Main activity")

if __name__ == "__main__":
	main()
