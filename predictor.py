# -*- coding: utf-8 -*-
All right reserved to Louann Naccache
"""

import yfinance as yf

# Télécharger les données d'Apple (AAPL)
ticker = yf.Ticker("AAPL")

# Récupérer l'historique des prix (par défaut : 1 mois)
data = ticker.history(start="2022-01-01", end="2025-07-31")

# Afficher uniquement la colonne 'Close'
print(data['Close'])

import numpy as np


# on peut faire aussi np.mean
# max est une mauvaise idée

length = 10

my_indicator = []
means = []
medians = []
for i in range(length,len(data['Close'])):
  median = np.median(data['Close'][i-length:i])
  mean = np.median(data['Close'][i-length:i])
  result = mean - median
  my_indicator.append(result)
  means.append(mean)
  medians.append(median)

data = data['Close']







data

xs = []
ys = []



data

len(data),len(medians),len(means),len(my_indicator)

dataNew = [data[i] for i in range(len(data)-length)]

len(dataNew),len(medians),len(means),len(my_indicator)

len(dataNew) == len(medians) == len(means) == len(my_indicator)

for i in range(length,len(dataNew)):
  local_mean = np.mean(dataNew[i-length:i-1])
  if True:
    x = np.array(dataNew[i-length:i-1] - local_mean)
    x1 = np.array(medians[i-length:i-1])
    x2 = np.array(means[i-length:i-1])
    x3 = np.array(my_indicator[i-length:i-1])
    x_ = np.array([x,x1,x2,x3])
    #print(x,x1.shape,x2.shape,x3.shape,x_.shape)
    xs.append(x_)

  y = np.array([dataNew[i]])
  ys.append(y)

xs,ys

xs = np.array(xs)
ys = np.array(ys)

xs

xs = np.reshape(xs,(xs.shape[0],length-1,4))
"""
ys2 = [0. for i in range(ys.shape[0]*(length-1)*4)]
ys3 = np.array([ys2])
ys4 = np.reshape(ys3, (ys.shape[0],length-1,4))
xs.shape,ys.shape
"""
#xs = np.reshape(xs,(xs.shape[0],4,length-1))
#xs = np.reshape(xs,(xs.shape[0],length-1))

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense,Dropout, LSTM

import tensorflow as tf

ys.shape

xs[0][0]

xs[0][0]

xs = np.reshape(xs,(xs.shape[0],1,9*4))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(np.array(xs).reshape(-1, 9*4))
scaled_data2 = scaler.fit_transform(np.array(ys).reshape(-1, 1))
scaled_data.shape

scaled_data = np.reshape(scaled_data,(scaled_data.shape[0],1,36))

model = Sequential()
model.add(LSTM(600))
model.add(Dense(1))
model.compile(loss='mae')

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(LSTM(300, activation='relu', return_sequences=True))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='sgd', loss='mae',metrics=['accuracy','mae'])
y_pred_scaled = model.predict(X)
y_pred = scaler.inverse_transform(y_pred_scaled)
#model.build(input_shape=(None,xs.shape[1]))
model.summary()"""

model.weights

model.fit(scaled_data,scaled_data2,epochs=10)
"""
for i in range(len(xs)-100,len(xs)):
  prediction = model.predict(np.array([xs[i]]))[0]
  print(prediction)
  y_pred.append(prediction)

  y_true.append(ys[i][0] + mean)

y_true, y_pred

y_true[-1], y_pred[-1],y_true[-1] - y_pred[-1]

y_pred

# Visualiser les résultats
plt.figure(figsize=(10, 6))
plt.plot(y_true, scalex=True,label='Prix réel')
plt.plot(y_pred[0], scalex=True,label='Prix prédit')
plt.plot(y_pred[0], scalex=True,label='Prix prédit')
plt.plot(y_pred[1], scalex=True,label='Prix prédit')
plt.plot(y_pred[2], scalex=True,label='Prix prédit')
plt.plot(y_pred[3], scalex=True,label='Prix prédit')
plt.legend()
plt.title("Prédiction du cours AAPL avec RNN")
plt.show()
"""

y_pred_scaled = model.predict(np.array([scaled_data[-1]]))
y_pred = scaler.inverse_transform(np.array(y_pred_scaled))
y_true = scaler.inverse_transform(np.array([scaled_data2[-1]]))

print("--------------------------")
print("y_true: ",y_true," / y_pred: ", y_pred)

"""

# Evaluate graphically the model

yy_true = []
yy_pred = []

for i in range(100):
  
  y_pred_scaled = model.predict(np.array([scaled_data[i]]))
  y_pred = scaler.inverse_transform(np.array(y_pred_scaled))
  yy_pred.append(y_pred[0][0])
  
  y_true = scaler.inverse_transform(np.array([scaled_data2[i]]))
  yy_true.append(y_true[0][0])

  print("y_true: ",y_true," / y_pred: ", y_pred)
print("--------------------------")
print("y_true: ",y_true," / y_pred: ", y_pred)
import matplotlib.pyplot as plt
plt.plot(yy_true, scalex=True,label='Prix réelle')
plt.plot(yy_pred, scalex=True,label='Prix prédit')
plt.show()

"""

# Predict the future price of APPL


yy_pred = []
yy_true = []

for i in range(800,0,-1):
  
  y_pred_scaled = model.predict(np.array([scaled_data[-i]]))
  y_pred = scaler.inverse_transform(np.array(y_pred_scaled))
  yy_pred.append(y_pred[0][0])
  
  y_true = scaler.inverse_transform(np.array([scaled_data2[-i]]))
  yy_true.append(y_true[0][0])

  print("y_true: ",y_true," / y_pred: ", y_pred)

for i in range(100):

  length = 10
  my_indicator = []
  means = []
  medians = []
  for i in range(length,len(yy_pred)):
    median = np.median(yy_pred[i-length:i])
    mean = np.median(yy_pred[i-length:i])
    result = mean - median
    my_indicator.append(result)
    means.append(mean)
    medians.append(median)

  data = yy_pred

  xs = []
  ys = []


  dataNew = [data[i] for i in range(len(data)-length)]


  for i in range(length,len(dataNew)):
    local_mean = np.mean(dataNew[i-length:i-1])
    if True:
      x = np.array(dataNew[i-length:i-1] - local_mean)
      x1 = np.array(medians[i-length:i-1])
      x2 = np.array(means[i-length:i-1])
      x3 = np.array(my_indicator[i-length:i-1])
      x_ = np.array([x,x1,x2,x3])
      #print(x,x1.shape,x2.shape,x3.shape,x_.shape)
      xs.append(x_)

    y = np.array([dataNew[i]])
    ys.append(y)

  xs = np.array(xs)
  ys = np.array(ys)

  xs = np.reshape(xs,(xs.shape[0],1,9*4))


  scaled_data = scaler.fit_transform(np.array(xs).reshape(-1, 9*4))
  scaled_data2 = scaler.fit_transform(np.array(ys).reshape(-1, 1))
  scaled_data.shape

  scaled_data = np.reshape(scaled_data,(scaled_data.shape[0],1,36))

  y_pred_scaled = model.predict(np.array([scaled_data[-1]]))
  y_pred = scaler.inverse_transform(np.array(y_pred_scaled))
  yy_pred.append(y_pred[0][0])

  
  print("--------------------------")
  print("y_true: ",y_true," / y_pred: ", y_pred)
  
plt.plot(yy_true, scalex=True,label='Prix réelle')
plt.plot(yy_pred, scalex=True,label='Prix prédit')
plt.show()
