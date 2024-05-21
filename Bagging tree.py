# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 22:38:49 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

T = 100
x_axis = np.linspace(0, 2*np.pi, T) #Т - число точек, 2пи - число за которое синусоида возвращается в ту же точку
y_axis = np.sin(x_axis) # функция Target - синусоида

N = 30 # генерируем 30 результатов из оригинальной функции.
idx = np.random.choice(T, size = N, replace = False) # у нас x_axis array size 0:T. Семплим оттуда 30 разных точек.
Xtrain = x_axis[idx].reshape(N,1) # насемплили, зарешейпили, так как наша дрянь требует другую размерность
Ytrain = y_axis[idx] # по аналогичным индексам тянем Y
model = DecisionTreeRegressor() # модель дерева решений
model.fit(Xtrain, Ytrain)
prediction = model.predict(x_axis.reshape(T, 1))
print("score for 1 tree", model.score(x_axis.reshape(T, 1), y_axis))

plt.plot(x_axis, prediction)
plt.plot(x_axis, y_axis)
plt.show()

class BaggedTreeRegressor:
  def __init__(self, B):
    self.B = B #Количество моделей в ансамбле

  def fit(self, X, Y):
    self.models = [] #Создаем пустой лист, куда будем загружать модели
    for b in range(self.B): 
      idx = np.random.choice(N, size=N, replace = True) #Список индексов, которые будем вытаскивать.
      Xb = X[idx] 
      Yb = Y[idx]

      model = DecisionTreeRegressor() #Создали модельку
      model.fit(Xb, Yb) #Фитанулись
      self.models.append(model) #Накинули модельку в список

  def predict(self, X): 
    predictions = np.zeros(len(X)) #make list of predictions length x_test
    for model in self.models: #for each model in list
      predictions += model.predict(X) #make each model predict and add to array
    return predictions / self.B #array of mean probabilities for each x_test

  def score(self, X, Y): # 
    d1 = Y - self.predict(X)
    d2 = Y - Y.mean()
    return 1 - d1.dot(d1) / d2.dot(d2)

model = BaggedTreeRegressor(200)
model.fit(Xtrain, Ytrain)
print("score for bagged trees:", model.score(x_axis.reshape(T, 1), y_axis))
prediction = model.predict(x_axis.reshape(T, 1))

plt.plot(x_axis, prediction)
plt.plot(x_axis, y_axis)
plt.show()