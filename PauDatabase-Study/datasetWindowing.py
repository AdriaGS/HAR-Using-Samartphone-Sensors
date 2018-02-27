import preprocessing as pp
import pandas as pd
import numpy as np
from sklearn import preprocessing
import csv
import pickle
import scipy.stats

path_root = '/Users/adriagil/UNI/IPAL/MobilityApps/Activitrack_adria/Database/'
users = ['pau', 'joaquim', 'alena', 'viet-thi']
activities = ['Walk', 'Bike', 'Train', 'Bus', 'Car', 'Nothing'] # 'Run',
NS_TO_MS = 0.000001
minWinSize = 10
windows = pd.DataFrame()

def build_windowed_collection(df_user_activity, features_to_expand, window_length):
	list_all_windowed = []
	dfw_user_activity = dict.fromkeys(df_user_activity) #df_user_activity.copy()
	for u, lact in list(df_user_activity.items()):
		dfw_user_activity[u] = dict.fromkeys(lact)
		for a, ldf in list(lact.items()):
			dfw_user_activity[u][a] = []
			i = 0
			for df, usr, act, f in ldf:
				#print(df.head())
				i = i + 1
				windowed = build_windowed_df(df, window_length, features_to_expand)
				if(len(windowed) > minWinSize):
					windowed['label'] = act
					dfw_user_activity[u][a].append(windowed)
					list_all_windowed.append(windowed)
				else:
					print("Too few data for a window: ", str(i))

	return list_all_windowed

def build_windowed_df(df, t, features_to_expand):
	
	fun_to_apply_td = {
		'mean_td': lambda x: x.mean(),
		'median_td': lambda x: x.median(),
		'std_td': lambda x: x.std(),
		'min_td': lambda x: x.min(),
		'max_td': lambda x: x.max(),
		'0.25_td': lambda x: x.quantile(0.25),
		'0.50_td': lambda x: x.quantile(0.50),
		'0.75_td': lambda x: x.quantile(0.75)
	}
	
	df.timestamp = df.timestamp.astype('datetime64[ns]')
	dfr = df.sort_values(by='timestamp').rolling(window=t, on='timestamp')
	
	dictDF = {}
	dictDF['count'] = dfr['accelerometer_x'].count()

	if(len(dictDF['count']) > minWinSize):
		dictDF = ({col + '_' + fname : f(dfr[col]) for col in features_to_expand for fname, f in fun_to_apply_td.items()})
	else:
		print("Too few data")
	
	return pd.DataFrame.from_dict(dictDF)

def main():
	
	with open('objs.pkl', 'rb') as f:
		df_user_activity, f2 = pickle.load(f)

	#Windowed Dataframe

	xr = pd.concat(build_windowed_collection(df_user_activity, f2, '3s'), ignore_index=True)

	CDF = xr.copy() #.dropna(subset=features)
	Xw = CDF.drop('label', axis=1)
	yw = CDF['label']
	Xw = Xw.fillna(method='ffill')
	Xw = Xw.fillna(method='bfill') 

	with open('datasetWin.pkl', 'wb') as f:
		pickle.dump([Xw, yw, activities, list(f2)], f)


if __name__ == '__main__':
	main()

