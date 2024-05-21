# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 13:15:33 2023

@author: AMD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
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
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    
    plt.scatter(Y_test, predictions)
    plt.xlabel("target")
    plt.ylabel("prediction")
    ymin = np.round(min(min(Y_test), min(predictions)))
    ymax = np.round(max(max(Y_test), max(predictions)))
    print("ymin:", ymin, "ymax", ymax)
    r = range(int(ymin), int(ymax) + 1)
    plt.plot(r, r)
    plt.show()
    
    plt.plot(Y_test, label='targets')
    plt.plot(predictions, label = 'predictions')
    plt.legend()
    plt.show()
    
    baseline = LinearRegression()
    single_tree = DecisionTreeRegressor()
    print("CV single tree:", cross_val_score(single_tree, X_train, Y_train).mean())
    print("CV baseline:", cross_val_score(baseline, X_train, Y_train).mean())
    print("CV forest:", cross_val_score(model, X_train, Y_train).mean())
    
    single_tree.fit(X_train, Y_train)
    baseline.fit(X_train, Y_train)
    print("test score single tree:", single_tree.score(X_test, Y_test))
    print("test score baseline:", baseline.score(X_test, Y_test))
    print("test score forest:", model.score(X_test, Y_test))
    
    
    