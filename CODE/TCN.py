# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:04:40 2021

@author: LHY
"""


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd

#from keras.models import Sequential
#from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def read_pkl(param):
    path = "C:/Users/LHY/Documents/WeChat Files/wxid_nhchl6d5yu9a22/FileStorage/File/2021-12/input/" + param + ".pkl"
    df=open(path,'rb')
    data3=pickle.load(df)
    return data3

X_train = []
x_1 = read_pkl("google_trend")
#x_1 = [(i-min(x_1))/(max(x_1)-min(x_1)) for i in x_1]
X_train.append(x_1)
x_2 = read_pkl("polarity")
X_train.append(x_2)
X_train = np.array(X_train)
X_train = X_train.T

Y_train = []
y_1 = read_pkl("new_cases")
y_1 = [(i-min(y_1))/(max(y_1)-min(y_1)) for i in y_1]
Y_train.append(y_1)
y_2 = read_pkl("diff")
y_2 = [(i-min(y_2))/(max(y_2)-min(y_2)) for i in y_2]
Y_train.append(y_2)
Y_train = np.array(Y_train)
Y_train = Y_train.T





batch_size, time_steps, input_dim = None, 20, 1
tcn_layer = TCN(input_shape=(time_steps, input_dim))
m = Sequential([
    tcn_layer,
    Dense(1)
])

m.compile(optimizer='adam', loss='mse')

tcn_full_summary(m, expand_residual_blocks=False)


#m.fit(X_train[0:30,0:1], Y_train[0:30,0:1], epochs=2, validation_split=0.2)
#forecast = m.predict(X_train[30:31,0:1])
#print(forecast)


y_pred_linear = []
y_actual_linear = []
for i in range(91): 
     m.fit(X_train[0:30+i,0:2], Y_train[0:30+i,0:1], epochs=500, validation_split=0.2)
 
     y_pred = m.predict(X_train[31+i:32+i,0:2])
     y_pred_linear.append(y_pred[0])
     y_actual_linear.append(Y_train[31+i:32+i,0:1][0])


plt.plot(np.array(y_pred_linear), label="predict")
plt.plot(y_actual_linear,label = 'actual')
plt.legend()




