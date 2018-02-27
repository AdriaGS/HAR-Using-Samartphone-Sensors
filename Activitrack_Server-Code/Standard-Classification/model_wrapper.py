import pandas as pd
import pickle
import optparse
import preprocessing as pp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import tree
from sklearn import preprocessing

class ModelWrapper:
    '''
    Manage and use an activity classifier.
    '''

    _DEFAULT_MODEL_NAME = 'default.pck'
    _FEATURES_TO_USE = ['gyroscope_x', 'gyroscope_y',
           'gyroscope_z', 'gyroscope_uncalibrated_x', 'gyroscope_uncalibrated_y',
           'gyroscope_uncalibrated_z', 'gyroscope_uncalibrated_x2',
           'gyroscope_uncalibrated_y2', 'gyroscope_uncalibrated_z2', 'gravity_x',
           'gravity_y', 'gravity_z', 'accelerometer_x', 'accelerometer_y',
           'accelerometer_z', 'linear_acceleration_x', 'linear_acceleration_y',
           'linear_acceleration_z', 'magnetic_field_x', 'magnetic_field_y',
           'magnetic_field_z', 'magnetic_field_uncalibrated_x',
           'magnetic_field_uncalibrated_y', 'magnetic_field_uncalibrated_z',
           'magnetic_field_uncalibrated_x2', 'magnetic_field_uncalibrated_y2',
           'magnetic_field_uncalibrated_z2', 'rotation_vector_x',
           'rotation_vector_y', 'rotation_vector_z', 'rotation_vector_x2',
           'rotation_vector_y2', 'geomagnetic_rotation_vector_x',
           'geomagnetic_rotation_vector_y', 'geomagnetic_rotation_vector_z',
           'geomagnetic_rotation_vector_x2', 'geomagnetic_rotation_vector_y2']


    def __init__(self):
        self.__loaded_model = None
        pass


    def load_default_model(self):
        print("Not implemented")
        return None

    def _build_df_collection(self, path_to_training_data, files_by_user, activities, users, all_files):
        '''
        Return structures containing the dataframes, per user and activity; and the features present in all dataframes.
        '''

        files_user_activity = {u: {a: [f for f in files_by_user[u] if a in f] for a in activities} for u in users}
        activity_map = {a: [f for f in all_files if a in f] for a in activities}

        features_intersection = {}
        df_user_activity = files_user_activity.copy()
        df_activities = {}
        for u, ul in files_user_activity.items():
            for a, fl in ul.items():
                if not a in df_activities:
                    df_activities[a] = []
                df_user_activity[u][a] = []
                for f in fl:
                    df = pd.read_csv(path_to_training_data + u + '/' + f, na_values='null').iloc[4:-4]
                    df_activities[a].append((df, u, a, f))
                    df_user_activity[u][a].append((df, u, a, f))
                    try:
                        pp.post_process_df(df, feedback=False)
                        if len(features_intersection) == 0:
                            features_intersection = set(df.columns)
                        else:
                            features_intersection.intersection_update(set(df.columns))
                        df['file'] = f
                        df['user'] = u
                    except Exception as e:
                        print('Error at {}, {}: {}'.format(u, a, f))
                        print(df.shape)
                        raise e

        features_intersection.remove('label')
        return df_user_activity, df_activities, features_intersection


    def _build_matrix(self, features_intersection, df_activities):
        '''
        Take the merged activity dict and return X, y and info.
        '''
        df_activities_merged = {a: pd.concat(map(lambda x: x[0], df_list), ignore_index=True) for a, df_list in df_activities.items() if len(df_list) > 0}

        complete_df = pd.concat(list(df_activities_merged.values()), ignore_index=True)

        feat = list(features_intersection.intersection(self._FEATURES_TO_USE))

        CDF = complete_df.copy()
        X = CDF.drop('label', axis=1)
        info = CDF[['_id', 'file', 'user']]
        y = CDF['label']

        X = X[feat]
        return X, y, info

    def _store_model(self, flist, clf, destination_path):
        '''
        Export the classifier to the destination path passed.
        '''
        object_to_store = {
            'features': flist,
            'model': clf
        }

        with open(destination_path, 'wb') as f:
            pickle.dump(object_to_store, f)


    def train_and_export(self, path_to_training_data, export_file_path, users_wanted=[]):
        '''
        Train the model and export the trained model to export_file_name
        '''
        if self.__loaded_model == None:
            print("No model loaded! Call load_model to load one first")
            raise Exception("No model loaded")

        print("Train and Export selected.")
        files_by_user, all_files = pp.import_all_files(path_to_training_data)
        print("We have data from %d users: " % len(files_by_user), list(files_by_user.keys()))
        print("We have %d files in total. " % len(all_files))

        activities_to_use = ['Walk', 'Bike', 'Train', 'Bus', 'Car', 'Nothing'] # 'Run',

        if len(users_wanted) == 0:
            users_wanted = files_by_user.keys()

        print("Building collection")

        df_user_activity, df_activities, features_intersection = self._build_df_collection(path_to_training_data, files_by_user, activities_to_use, users_wanted, all_files)

        print("Collection built")

        X, y, info = self._build_matrix(features_intersection, df_activities)
        print("Matrix built")
        print(X.head())
        print(y.head())
        feat, clf = self.__loaded_model.values()
        print("Model loaded:")
        print(feat, clf)

        X = X.fillna(method='ffill')
        X = X.fillna(method='bfill')
        X_train, X_test, y_train, y_test = train_test_split(X[feat], y, stratify=y, test_size=0.20)

        print("We will fit the clf now")
        clf.fit(X_train, y_train)

        print("We will compute its score now")
        score = clf.score(X_test, y_test)
        print("Done training. Exporting the model now.")
        self._store_model(feat, clf, export_file_path)
        print("Exported.")
        return score


    def load_model(self, path):
        '''
        Load the model stored in the path passed.
        This method needs to be called before trying to predict or train.
        '''
        with open(path, 'rb') as f:
            self.__loaded_model = pickle.load(f)
            #print(self.__loaded_model)
        return self.__loaded_model


    def _check_model_loaded(self):
        '''
        Check if there is a model loaded and load the default one if there isn't one.
        '''
        if self.__loaded_model == None:
            print("No model loaded! Call load_model(path) first.")
            # print("No model loaded! Using the default one")
            self.__loaded_model = load_default_model()


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

    def get_scaled(self, df):
        x = df.values #returns a numpy array
        x_scaled = preprocessing.StandardScaler().fit_transform(x)
        df_s = pd.DataFrame(x_scaled)
        return df_s


    def predict_csv(self, csv_file):
        '''
        Predict the activity label(s) for the row(s) of the passed CSV file.
        '''
        self._check_model_loaded()
        df = pd.read_csv(csv_file, na_values='null')
        df = df.fillna('0.0;0.0;0.0')
        df = df.drop(df[(df["android.sensor.gyroscope_uncalibrated"] == 0) | (df["android.sensor.gravity"] == 0) | (df["android.sensor.accelerometer"] == 0)].index)
        #print(df)
        value_dic = self.__loaded_model
        feat = value_dic['feat']
        clf = value_dic['clf']
        #TODO: make sure that the timestamp field is not the phone time measure
        #print("feat", feat)
        df = pp.post_process_df(df)#[feat]
        missing_cols = [c for c in feat if not c in df.columns]
        for c in missing_cols:
            df[c] = 0
        #df[missing_cols] = np.nan
        #print(df.columns)
        ts = df.timestamp
        df_s = self.get_scaled(df[feat])
        prediction =  clf.predict(df_s)
        #print(zip(ts, prediction)[:5])
        return prediction


def main():
    '''
    p = optparse.OptionParser()
    p.add_option('--predict', '-p', default="placeholder")
    options, arguments = p.parse_args()

    predict_path = options.predict
    print("Going to predict the activities for the file %s" % predict_path)
    '''
    mw = ModelWrapper()
    file_path = '/home/pau/Documents/activitrack/code/knn_3_001.pickle'
    features, clf = mw.load_model(file_path)
    #load_model(file_path)
    print("Model loaded!")
    print(features, clf)

    path_csv_to_predict = "/home/pau/Dropbox/Singapore/activitrack/to_predict/example_000.csv"
    df = pd.read_csv(path_csv_to_predict, na_values='null')
    res = mw.predict_csv(path_csv_to_predict)
    print(res)

    print("Training...")
    path_to_training_data = "/home/pau/Dropbox/Singapore/activitrack/"
    res = mw.train_and_export(path_to_training_data, "kekerino.pickle")
    print(res)

if __name__ == "__main__":
    main()
