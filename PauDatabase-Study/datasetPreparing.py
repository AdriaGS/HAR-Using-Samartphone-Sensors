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
windows = pd.DataFrame()

def build_df_collection(files_by_user, all_files):
	'''
	returns df_user_activity, df_activities, details_df, features_intersection
	'''
	files_user_activity = {u: {a: [f for f in files_by_user[u] if a in f] for a in activities} for u in users}
	activity_map = {a: [f for f in all_files if a in f] for a in activities}

	features_intersection = {}
	df_user_activity = files_user_activity.copy()
	df_activities = {}
	details_df = {}
	for u, ul in files_user_activity.items():
		for a, fl in ul.items():
			if not a in df_activities:
				df_activities[a] = []
			df_user_activity[u][a] = []
			for f in fl:
				df = pd.read_csv(path_root + u + '/' + f, na_values='null').iloc[4:-4]
				df_activities[a].append((df, u, a, f))
				df_user_activity[u][a].append((df, u, a, f))
				#details_df[df] = (u, a, f)
				try:
					pp.post_process_df(df, feedback=False)
					#print("processed")
					if len(features_intersection) == 0:
						features_intersection = set(df.columns)
					else:
						features_intersection.intersection_update(set(df.columns))
					df['file'] = f
					df['user'] = u
				except Exception as e:
					print('Error at {}, {}: {}'.format(u, a, f))
					print(df.shape)
					#print(df)
					break

	features_intersection.remove('label')
	return df_user_activity, df_activities, details_df, features_intersection



def build_matrix(features, df):
	CDF = df.copy() #.dropna(subset=features)
	X = CDF.drop('label', axis=1)
	info = CDF[['_id', 'file', 'user']]
	y = CDF['label']
	
	X = X[features]
	return X, y, info

def get_scaled_X(X):
	Xsc = X.copy()
	Xsc = preprocessing.StandardScaler().fit_transform(Xsc)

	Xres = pd.DataFrame(data=Xsc, columns=X.columns)
	return Xres

def main():
	
	files_by_user, all_files = pp.import_all_files(path_root)

	print("We have data from %d users. " % len(files_by_user))#, list(files_by_user.keys()))
	print("We have %d files in total. " % len(all_files))

	df_user_activity, df_activities, details_df, features_intersection = build_df_collection(files_by_user, all_files)

	#print(sum([len(l) for l in df_user_activity['pau'].values()]))
	#print(len(files_by_user['pau']))
	#print(features_intersection)

	df_activities_merged = {a: pd.concat(map(lambda x: x[0], df_list), ignore_index=True) for a, df_list in df_activities.items() if len(df_list) > 0}
	complete_df = pd.concat(list(df_activities_merged.values()), ignore_index=True)

	features_to_use = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z', 'gravity_x', 'gravity_y', 'gravity_z', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z',  'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z', 'rotation_vector_x', 'rotation_vector_y', 'rotation_vector_z', 'geomagnetic_rotation_vector_x', 'geomagnetic_rotation_vector_y', 'geomagnetic_rotation_vector_z']

	print(complete_df.groupby(by="user").label.value_counts())

	f1 = features_to_use
	f2 = features_intersection.intersection(features_to_use)
	
	print(list(f2))
	
	X, y, info = build_matrix(list(f2), complete_df)

	print("Using %d features." % X.shape[1])
	print("We have %d data points." % X.shape[0])
	pd.value_counts(y)

	X = X.fillna(method='ffill')
	Xs = get_scaled_X(X)

	with open('objs.pkl', 'wb') as f:
		pickle.dump([df_user_activity, f2], f)

	with open('datasetNoWin.pkl', 'wb') as f:
		pickle.dump([Xs, y, activities, list(f2)], f)

	with open('dataset.pkl', 'wb') as f:
		pickle.dump([Xs, y, list(f2)], f)


if __name__ == '__main__':
	main()

