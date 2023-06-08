import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import glob
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures  # Класс преобразователь
from sklearn import metrics
import pickle
#загрузка модели
filename = 'poly_linear_model.sav'
load_model = pickle.load(open(filename, 'rb'))
df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("test", "test*.csv"))), ignore_index= True)#загрузка всех тестовых данных в одну таблицу
X_test=df['X_new']
y_test= df['Y']
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_test.to_numpy().reshape(-1, 1))       # Преобразуем данные
y_pr=load_model.predict(X_poly)
print('среднеквадраточное отклонение : ', np.sqrt(metrics.mean_squared_error(y_test, y_pr)))
f = open('score.txt','w')  # открытие в режиме записи
f.write('среднеквадраточное отклонение : {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pr))))  # запись 
f.close()  # закрытие файла
