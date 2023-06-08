# -*- coding: utf-8 -*-
"""model_preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11Ji_4f-xPpIwu-RyE-RCuzl2uFOG7plh
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def lenfile(path):
    count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    return count

def scale(train_file,test_file):
    scale=StandardScaler()
    train_df=pd.read_csv(train_file)
    test_df=pd.read_csv(test_file)
    train=train_df["X"].to_numpy().reshape(-1, 1)
    scale.fit(train)
    X_train=scale.transform(train)
    X_test=scale.transform(test_df["X"].to_numpy().reshape(-1, 1))
    train_df['X_new']=X_train
    test_df['X_new']=X_test
    train_df.to_csv('train/train{}.csv'.format(i+a), index=False)
    return train_df, test_df

for i in range(lenfile('train')):
        train_df, test_df=scale('train/train{}.csv'.format(i),'test/test{}'.format(i))  
        train_df.to_csv('train/train{}.csv'.format(i), index=False)
        test_df.to_csv('test/test{}'.format(i), index=False)