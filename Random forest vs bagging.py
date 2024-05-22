# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 21:31:59 2023

@author: AMD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
Y = raw_df.values[1::2, 2]

for i in range(X.shape[1]):
    if i != 3:
        scaler = StandardScaler()
        X[:,i] = scaler.fit_transform(X[:,i].reshape(-1, 1)).flatten()

X, Y = shuffle(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

T = 100

test_error_rf = np.empty(T)
test_error_bag = np.empty(T)

for num_trees in range(T):
    if num_trees == 0:
        test_error_rf[num_trees] = None
        test_error_bag[num_trees] = None
    else:
        rf = RandomForestRegressor(n_estimators = num_trees)
        rf.fit(X_train, Y_train)
        test_error_rf[num_trees] = rf.score(X_test, Y_test)
        
        bg = BaggingRegressor(n_estimators = num_trees)
        bg.fit(X_train, Y_train)
        test_error_bag[num_trees] = bg.score(X_test, Y_test)
        
    if num_trees % 10 == 0:
        print('num_trees:', num_trees)
        plt.plot(test_error_rf, label="rf")
        plt.plot(test_error_bag, label='bag')
        plt.legend()
        plt.show()
