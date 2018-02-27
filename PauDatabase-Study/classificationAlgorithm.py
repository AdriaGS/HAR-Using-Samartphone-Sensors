import preprocessing as pp
import pandas as pd
import numpy as np
from sklearn import preprocessing
import csv
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import cross_val_score
import time

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import sys

def get_model_name(clf):
	return str(clf.__class__).split(".")[-1].split("'")[0]

def get_timed_CV_score(clf, X_model, y_model, skf, feedback=True):
	clf_name = get_model_name(clf)
	#print("%s: computing cv score..." % clf_name)
	t1 = time.time()
	scores = cross_val_score(clf, X_model, y_model, cv=skf)
	t2 = time.time()
	return t2-t1, scores

def main():
	with open('datasetWin.pkl', 'rb') as f:
		Xw, y, activities, features = pickle.load(f)

	X_model, X_validate, y_model, y_validate = train_test_split(Xw, y, test_size=0.20, random_state=42, stratify=y, shuffle=True)

	#Classification Algorithms

	list_models_to_eval = [
		KNeighborsClassifier(n_neighbors=3),
		KNeighborsClassifier(n_neighbors=4),
		KNeighborsClassifier(n_neighbors=5),
		tree.DecisionTreeClassifier(),
		]

	skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=4)
	models_evaluation = {clf: get_timed_CV_score(clf, X_model, y_model, skf) for clf in list_models_to_eval}

	for m, res in models_evaluation.items():
		t, scores = res
		print(get_model_name(m), "%.2f (std %.2f)" % (scores.mean()*100, scores.std()*100) + ' %.2f' % t)

	chosen_model = KNeighborsClassifier(n_neighbors = 3)
	chosen_model.fit(X_model, y_model)
	score = chosen_model.score(X_validate, y_validate)
	print("Accuracy on the validation data: %.2f" % (score*100))

	y_val_predicted = chosen_model.predict(X_validate)

	print(classification_report(y_validate, y_val_predicted))

	#Save model
	chosen_model.fit(Xw, y)

	object_to_store = {
		'feat': features,
		'clf': chosen_model
	}

	with open('kNN3_001.pickle', 'wb') as f:
		pickle.dump(object_to_store, f, protocol=pickle.HIGHEST_PROTOCOL)
		pass

if __name__ == '__main__':
	main()