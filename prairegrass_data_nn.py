# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:16:52 2020

@author: k_sha
"""



#check versions
import sys
import keras
import tensorflow as tf

print('Python Version', sys.version)
print('Keras Version', keras.__version__)
print('TF Version', tf.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

from pandas import read_csv
#from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
from scipy.stats import pearsonr
from keras.utils import plot_model
from joblib import dump, load

# fix random seed for reproducibility
seed = np.random.seed(7)

#load data
load_data = read_csv("pg_ml_data_stable.csv")
load_data.shape
load_data.columns


#drop first four columns - these are not features
#drop 'Ustar (m/s)','HeatFlux (W/m2)', 'Zmix (m)',
# load_data = load_data[['Q (g/s)', 'Hs (m)', 'U_1m  (m/s)', 'T (C)', 'L (m)', 'stable', 'category_A', 'category_B', 'category_C',
#        'category_D', 'category_E', 'category_F', 'X', 'Y', 'Conc_MG']] 

# load_data.dtypes
# load_data.shape

#Standard approach to scale/transform on a column basis
array = load_data.values
X = array[:,:-1]

Y = array[:,-1]
# Y = Y * 1000
# Y = np.log10(Y)

X.dtype
Y.dtype

# split into 90% for train and 10% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=seed) 


y_test.shape



#drop column 5 stability class prior to training
#X_train = X_train[:,[0,1,2,3,4,6,7,8,9,10,11,12,13]]
#X_test = X_test[:,[0,1,2,3,4,6,7,8,9,10,11,12,13]]


X_train.dtype

#scale data between 0 and 1
#scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)


#Normalize data 
scaler = Normalizer().fit(X_train)

#save normalizer
dump(scaler, 'scaler_nn.joblib') 

#Generate a sample of X_test prior to transform to use in the API

col_names = ['Q (g/s)', 'Hs (m)', 'U_1m  (m/s)', 'Ustar (m/s)', 'HeatFlux (W/m2)',
       'Zmix (m)', 'T (C)', 'Wstar (m/s)', 'L (m)', 'stable', 'category_A',
       'category_B', 'category_C', 'category_D', 'category_E', 'category_F',
       'X', 'Y']

sample_1 = X_test[0]

dict(zip(col_names, sample_1))

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# y_train.shape
# y_train = y_train.reshape(-1,1)
# y_train.shape

# y_test.shape
# y_test = y_test.reshape(-1,1)
# y_test.shape



# scalery = MinMaxScaler(feature_range=(0, 1)).fit(y_train)
# y_train = scalery.transform(y_train)
# y_test = scalery.transform(y_test)

#Alternate approach to scale/transform on a column basis - not needed now
#NB Dir and RecNum have been dropped
# X = load_data[['Q (g/s)', 'Hs (m)', 'U_1m  (m/s)', 'Ustar (m/s)',
#        'HeatFlux (W/m2)', 'Zmix (m)', 'T (C)', 'L (m)', 'StabilityClass', 'X', 'Y']] 
# X.shape
# Y = load_data[['Conc_MG']]
# Y.shape

# # split into 67% for train and 33% for test
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed) 


# #scale/transform on a column basis - neat
# mapper = DataFrameMapper([
#     (['StabilityClass'], LabelEncoder()),
#     (['Q (g/s)', 'Hs (m)', 'U_1m  (m/s)', 'Ustar (m/s)',
#        'HeatFlux (W/m2)', 'Zmix (m)', 'T (C)', 'L (m)', 'X', 'Y'], MinMaxScaler(feature_range=(0, 1)))
#     ])

# scaler = mapper.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


# scalery = MinMaxScaler().fit(y_train)
# y_train = scalery.transform(y_train)
# y_test = scalery.transform(y_test)



# create model
model = Sequential()
model.add(Dense(18, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu', name='Dense_1'))
model.add(Dense(1024, activation='relu', name='Dense_2'))
model.add(Dense(1024, activation='relu', name='Dense_3'))
model.add(Dense(1024, activation='relu', name='Dense_4'))
model.add(Dense(1024, activation='relu', name='Dense_5'))
model.add(Dense(1024, activation='relu', name='Dense_6'))
model.add(Dense(1, name='Dense_7'))

# compile model
#optimizers: adadelta, adagrad, adam, adamax, nadam, rmsprop, sgd
#model.compile(loss='mean_absolute_error', optimizer='adamax', metrics=['mean_absolute_error'])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])


# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=1000, batch_size=400)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1])) 

#Check scores for single observations
score_1 = model.predict(X_test)
score_1[10]
y_test[10]

y_train[1]


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#I know it can be improved...are these inputs good enough to predict the conc at a 
#particular location in space?


fig, ax = plt.subplots()
ax.scatter(y_test, score_1)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.set_title('NN Comparison of Measured/Predicted for transformed target units')
plt.show()


#converted version

def reversey(y):
    y = 10 ** y
    y = y/1000
    return y
#convert before plotting

score_1_convert = reversey(score_1)
y_test_convert = reversey(y_test)

fig, ax = plt.subplots()
ax.scatter(y_test_convert, score_1_convert)
ax.plot([y_test_convert.min(), y_test_convert.max()], [y_test_convert.min(), y_test_convert.max()], 'k--', lw=4)
ax.set_title('NN Comparison of Measured/Predicted for original target units')
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()





plt.plot(reversey(y_train[:600,]))
plt.plot(reversey(model.predict(X_train[:600,])))
plt.title("NN First 600 test cases")
plt.show()


plt.plot(y_train)
plt.plot(model.predict(X_train))
plt.title("NN All training cases")
plt.show()


#make new scorers

def fb_score(y_test, y_pred):
    
    #scorer for root mean squared error
    y_test = y_test.reshape(-1,1)
    
    result = (y_test.mean() - y_pred.mean()) / (0.5 * (y_test.mean() + y_pred.mean()))
    #scorer for fractional bias
    
    return result

def fs_score(y_test, y_pred):
    
    #scorer for fractional variance
    
    #reshape y_test to be consistent with y_pred
    y_test = y_test.reshape(-1,1)
    
    result = (y_test.std() - y_pred.std()) / (0.5 * (y_test.std() + y_pred.std()))
    
    return result

def rmse_score(y_test, y_pred):
    
    #scorer for root mean squared error
    y_test = y_test.reshape(-1,1)
    
    result = np.average((y_test - y_pred) ** 2, axis=0)
    
    return np.sqrt(result)


def mse_score(y_test, y_pred):
    
    #scorer for mean squared error
    #reshape y_test to be consistent with y_pred
    y_test = y_test.reshape(-1,1)
    
    result = np.average((y_test - y_pred) ** 2, axis=0)
    
    return result


def nmse_score(y_test, y_pred):
    
    #scorer for normalized mean squared error
    
    #reshape y_test to be consistent with y_pred
    y_test = y_test.reshape(-1,1)
    
    result = np.average((y_test - y_pred) ** 2, axis=0) / (y_test.mean() * y_pred.mean())
    
    return result

def cc_score(y_test, y_pred):
    
    #scorer for correlation coefficient, R
    
    #reshape y_test to be consistent with y_pred
    y_pred = y_pred.astype("float64").ravel()
        
    result = pearsonr(y_test, y_pred)
    #result = np.corrcoef(y_test, y_pred)
    
    return result

def fac2_score(y_test, y_pred):
    
    #scorer for fraction within a factor of 2
    
    #reshape y_test to be consistent with y_pred
    y_pred = y_pred.astype("float64").ravel()
    
    #print(y_pred.dtype)
    #print(y_test.dtype)
    
    
    compare = (y_pred / y_test)
    
    result = np.logical_and(0.5 <= compare, compare <= 2)
    
    #print(compare[:5], result[:5])
    
    result = np.count_nonzero(result)
        
    return round((result/len(y_test))*100)
    


#extract metrics
y_pred = model.predict(X_test)

MSE = mean_squared_error(y_test,y_pred)
print('MSE is:', MSE)
print('RMSE is:', math.sqrt(MSE))
#1267.978515191869
#rmse = 35.60868595149039 ~ very close to Azure ML
#mean_squared_error, r2_score, mean_absolute_error
r2 = r2_score(y_test,y_pred)
print('r2 is:', r2)
MAE = mean_absolute_error(y_test,y_pred)
print('MAE is:', MAE)


#new metrics
fb = fb_score(y_test, y_pred)
print('fb is:', fb)
fs = fs_score(y_test, y_pred)
print('fs is:', fs)
rmse = rmse_score(y_test, y_pred)
print('rmse is:', rmse)
mse = mse_score(y_test, y_pred)
print('mse is:', mse)
nmse = nmse_score(y_test, y_pred)
print('nmse is:', nmse)
cc = cc_score(y_test, y_pred)
print('cc is:', cc)
fac2 = fac2_score(y_test, y_pred)
print('fac2 is:', fac2, "%")



# #save the model
# from pickle import dump
# filename = "prairegrass_data_nn_model.sav"
# dump(model, open(filename, 'wb' ))


# #load the model from disk
# from pickle import load
# loaded_model = load(open(filename, 'rb'))

# #test loaded model
# plt.plot(y_train[:600,])
# plt.plot(loaded_model.predict(X_train[:600,]))
# plt.show()



#save final model
dump(model, 'model_nn.joblib') 


# loaded_model = load('filename.joblib') 

#test loaded model
# plt.plot(y_train[:600,])
# plt.plot(loaded_model.predict(X_train[:600,]))
# plt.show()


#plot the network

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)



#test output for the API
X_test[0]
X_test[0].reshape(1,-1).shape
y_test[0]
test = model.predict(X_test[0].reshape(1,-1))
test
reversey(test)


#test saved model
loaded_model = load('model_nn.joblib') 
test_loaded = loaded_model.predict(X_test[0].reshape(1,-1))
test_loaded
reversey(test_loaded)
#array([[7.738637]], dtype=float32)


