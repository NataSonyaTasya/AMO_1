# -*- coding: utf-8 -*-
"""model_preparation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qOxQ440KOt_AU4mVCosFYwTnw6wvavbM
"""
import pickle
import numpy as np
import pandas as pd
import os
import glob
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures  # Класс преобразователь

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("train", "train*.csv"))), ignore_index= True)
X_train=df['X_new']
y_train= df['Y']

pr = LinearRegression() # Полиномиальная регрессия

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train.to_numpy().reshape(-1, 1))       # Преобразуем данные

pr.fit(X_poly, y_train) # Обучаем полиномиальную регрессию
filename = 'poly_linear_model.sav'
pickle.dump(pr, open(filename, 'wb'))
