# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:23:18 2023

@author: AMD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.utils import shuffle

class AdaBoost:
    def __init__(self, M):
        self.M = M
    
    def fit(self, X, Y):
        self.models = []
        self.alphas = []
        
        N = X.shape[0]
        W = np.ones(N) / N
        
        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth = 1)
            tree.fit(X, Y, sample_weight = W)
            P = tree.predict(X)
            err = W.dot(P!=Y)
            alpha = 0.5*(np.log(1-err) - np.log(err))
            W = W * np.exp(-alpha*Y*P)
            W = W / W.sum()
            self.alphas.append(alpha)
            self.models.append(tree)
            
    def predict(self, X):
        N = X.shape[0]
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            FX += alpha * tree.predict(X)
        return np.sign(FX), FX
    
    def score(self, X, Y):
        P, FX = self.predict(X)
        L = np.exp(-Y*FX).mean()
        return np.mean(P == Y), L
        
if __name__ == '__main__':
    mnist = load_digits()
    X = pd.DataFrame(mnist.data).to_numpy(dtype='float32')
    Y = pd.DataFrame(mnist.target).to_numpy().flatten()
    X_train = X[:1500,:] / 255
    X_test = X[1500:,:] / 255
    Y_train = Y[:1500]
    Y_test = Y[1500:]
    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = shuffle(X_test, Y_test)
    
    idx_train = np.logical_or(Y_train == 1, Y_train == 0)
    idx_test = np.logical_or(Y_test == 1, Y_test == 0)
    
    X_train = X_train[idx_train]
    Y_train = Y_train[idx_train]
    X_test = X_test[idx_test]
    Y_test = Y_test[idx_test]
    
    Y_test = 2*Y_test - 1
    Y_train = 2*Y_train - 1
    
    T = 200
    
    train_errors = np.empty(T)
    test_losses = np.empty(T)
    test_errors = np.empty(T)
    
    for num_trees in range(T):
        if num_trees == 0:
            train_errors[num_trees] = None
            test_errors[num_trees] = None
            test_losses[num_trees] = None
            continue
        if num_trees % 20 == 0:
            print(num_trees)
            
        model = AdaBoost(num_trees)
        model.fit(X_train, Y_train)
        acc, loss = model.score(X_test, Y_test)
        acc_train, _ = model.score(X_train, Y_train)
        train_errors[num_trees] = 1 - acc_train
        test_errors[num_trees] = 1 - acc
        test_losses[num_trees] = loss
        
        if num_trees == T - 1:
            print("final train error:", 1 - acc_train)
            print("final test error:", 1 - acc)
    
    plt.plot(test_errors, label = 'test errors')
    plt.plot(test_losses, label = 'test losses')   
    plt.legend()
    plt.show()
    
    plt.plot(test_errors, label = 'test errors')
    plt.plot(train_errors, label = 'train losses')   
    plt.legend()
    plt.show()
