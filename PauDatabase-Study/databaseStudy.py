aimport preprocessing as pp
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA as sklearnPCA
import csv
import pickle
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt

path_root = '/Users/adriagil/UNI/IPAL/MobilityApps/Activitrack_adria/Database/'
users = ['pau', 'joaquim', 'alena', 'viet-thi']
activities = ['Walk', 'Bike', 'Train', 'Bus', 'Car', 'Nothing'] # 'Run',
NS_TO_MS = 0.000001
windows = pd.DataFrame()

def main():
	
	with open('dataset.pkl', 'rb') as f:
		xr, y, features = pickle.load(f)

	#y = xr.label
	#Xw = xr.drop('label',axis = 1 )
	Xw = xr
	Xw = Xw.fillna(method='ffill')
	Xw = Xw.fillna(method='bfill')

	#ax = sns.countplot(y, label="Count")
	#plt.show()
	Wk, Tr, Bs, Cr, Nt, Bk = y.value_counts()
	print('Number of Bike: ',Bk)
	print('Number of Bus: ',Bs)
	print('Number of Car: ',Cr)
	print('Number of Nothing: ',Nt)
	print('Number of Train: ',Tr)
	print('Number of Walk: ',Wk)

	pca = sklearnPCA(n_components=2) #2-dimensional PCA
	transformed = pd.DataFrame(pca.fit_transform(Xw))

	plt.scatter(transformed[y=='Walk'][0], transformed[y=='Walk'][1], label='Walk', c='darkgreen')
	plt.scatter(transformed[y=='Bike'][0], transformed[y=='Bike'][1], label='Bike', c='red')
	plt.scatter(transformed[y=='Train'][0], transformed[y=='Train'][1], label='Train', c='yellow')
	plt.scatter(transformed[y=='Bus'][0], transformed[y=='Bus'][1], label='Bus', c='blue')
	plt.scatter(transformed[y=='Car'][0], transformed[y=='Car'][1], label='Car', c='lightgreen')
	plt.scatter(transformed[y=='Nothing'][0], transformed[y=='Nothing'][1], label='Nothing', c='black')

	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()

