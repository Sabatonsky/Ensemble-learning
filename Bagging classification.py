# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 23:23:01 2023

@author: AMD
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def plot_decision_boundary(X, model):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Accent)

N = 500
D = 2
X = np.random.randn(N, D)
M = 100
sep = 2

X[:125] += np.array([sep, sep])
X[125:250]+= np.array([sep, -sep])
X[250:375] += np.array([-sep, -sep])
X[375:500] += np.array([-sep, sep])
Y = np.array([0]*125 + [1]*125 + [0]*125 + [1]*125)

model = DecisionTreeClassifier(max_depth = 5) # модель дерева решений
model.fit(X, Y)
print("score for one tree:", model.score(X, Y))
prediction = model.predict(X)

plot_decision_boundary(X, model)
plt.scatter(X[:,0], X[:,1], c = prediction)
plt.show()  

class BaggedTreeClassifier:
    def __init__(self, B):
        self.B = B
        
    def fit(self, X, Y):
        self.models = []
        
        for b in range(self.B):
            idx = np.random.choice(N, size=N, replace = True)
            Xb = X[idx]
            Yb = Y[idx]
            model = DecisionTreeClassifier(max_depth = 2)
            model.fit(Xb, Yb)
            self.models.append(model)
    
    def predict(self, X):
        prediction = np.zeros(X.shape[0])
        for model in self.models:
            prediction += model.predict(X)
        return np.round(prediction / self.B)
            
    def score(self, X, Y):
        P = self.predict(X)
        return  np.mean(Y == P)
    
model = BaggedTreeClassifier(200)
model.fit(X, Y)
print("score for bagged trees:", model.score(X, Y))
prediction = model.predict(X)

plot_decision_boundary(X, model)
plt.scatter(X[:,0], X[:,1], c = prediction)
plt.show()

plt.scatter(X[:,0], X[:,1], c = Y)
plt.show()