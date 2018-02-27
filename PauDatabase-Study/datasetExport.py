import preprocessing as pp
import pandas as pd
import numpy as np
from sklearn import preprocessing
import csv
import pickle

with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
	X, y, activities, features = pickle.load(f)

Xs = X.as_matrix()

print(np.shape(Xs))
print(features)

for a in activities:
	np.savetxt(a + ".csv", Xs[(np.where(y == a)[0])][:], delimiter=",")
	