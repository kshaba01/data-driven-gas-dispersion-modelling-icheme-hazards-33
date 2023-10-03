# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:20:17 2019

@author: k_sha
"""
#load required libraries


import os
import tensorflow as tf

#sci-kit learn keras doesnt work well with GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

#from pandas import read_csv
from matplotlib import pyplot 
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
#import xgboost as xgb
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
#import scorers
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from pandas import set_option
from pandas import read_csv
from sklearn.model_selection import train_test_split, ShuffleSplit
import pandas as pd
import math

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense


#new models as used by microsoft
from sklearn.linear_model import LassoLars
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor


# fix random seed for reproducibility
seed = np.random.seed(7)

#load data
load_data = read_csv("pg_ml_data_stable.csv")
load_data.shape
load_data.columns


# #drop first four columns - these are not features
# #drop 'Ustar (m/s)','HeatFlux (W/m2)', 'Zmix (m)',
# load_data = load_data[['Q (g/s)', 'Hs (m)', 'U_1m  (m/s)', 'T (C)', 'L (m)', 'category_A', 'category_B', 'category_C',
#        'category_D', 'category_E', 'category_F', 'X', 'Y', 'Conc_MG']] 

load_data.dtypes
load_data.shape

#Standard approach to scale/transform on a column basis
array = load_data.values
X = array[:,:-1]

Y = array[:,-1]
# Y = Y * 1000
# Y = np.log10(Y)

#Y = np.log(Y)

# split into 90% for train and 10% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=seed) 


#build NN model
def create_model():
    #create model
    model = Sequential()
    model.add(Dense(18, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1))
    #compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

# NNmodelin = KerasRegressor(build_fn=create_model, epochs=1000, batch_size=400, verbose = 0)



# NNmodel = pipeline.Pipeline([
#     ('rescale', StandardScaler()),
#     ('nn', NNmodelin())
# ])

# pipeline = pipeline.Pipeline([
#     ('clf',NNmodelin)
# ])

#history = NNmodel.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=1000, batch_size=400)

#prepare models
#tuples used to append two objects
#default model settings are used...
models = []
models.append(('NN', KerasRegressor(build_fn=create_model, epochs=1000, batch_size=400, verbose = 0)))
models.append(('LGBM', LGBMRegressor()))
models.append(('LR', make_pipeline(Normalizer(), LinearRegression()))) #loss = rss
models.append(('KNR', make_pipeline(Normalizer(), KNeighborsRegressor()))) #nearest neighbours
models.append(('DTR', make_pipeline(Normalizer(), DecisionTreeRegressor()))) #mse
models.append(('SVR', make_pipeline(Normalizer(), SVR())))
models.append(('XRT2', make_pipeline(Normalizer(), ExtraTreesRegressor(bootstrap=False, max_features=0.9500000000000001, min_samples_leaf=10, min_samples_split=16, n_estimators=100)))) #mse
models.append(('DRF', make_pipeline(Normalizer(), RandomForestRegressor(n_estimators=47, max_depth=None, min_samples_split=2)))) #mse
models.append(('XRT', make_pipeline(Normalizer(), ExtraTreesRegressor()))) #mse
models.append(('RFR', make_pipeline(Normalizer(), RandomForestRegressor()))) #mse
models.append(('LAS', make_pipeline(Normalizer(), LassoLars(alpha=.1))))
models.append(('GBR', make_pipeline(Normalizer(), GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls'))))

#extras
#models.append(('SGDR', SGDRegressor()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('XGB', xgb.XGBRegressor(max_depth=3, learning_rate=0.2,random_state=42, n_estimators=30, objective="reg:squarederror")))

             

#evaluate each model in turn...
results_mse = []
results_rmse = []
results_mae = []
results_r2 = []

results_mse_mean = []
results_rmse_mean = []
results_mae_mean = []
results_r2_mean = []



# results = []
# means = []
# stds =[]

model_names = []
#scoring = make_scorer(mean_squared_error,squared=False, greater_is_better=False)
#All scorer objects follow the convention that higher return values are better than lower return values. Thus metrics which measure the distance between the model and the data, like metrics.mean_squared_error, are available as neg_mean_squared_error which return the negated value of the metric.
scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']



# ttmse = []
# ttrmse = []
# ttr2 = []


cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=seed)

for model_name, model in models:
    
    #using cross_validate
    # scores = cross_validate(model, X, Y, cv=10, scoring=scoring)
    # results_mse.append(scores['test_neg_mean_squared_error'])
    # results_rmse.append(scores['test_neg_root_mean_squared_error'])
    # results_mae.append(scores['test_neg_mean_absolute_error'])
    # results_r2.append(scores['test_r2'])
    
    # results_mse_mean.append(scores['test_neg_mean_squared_error'].mean())
    # results_rmse_mean.append(scores['test_neg_root_mean_squared_error'].mean())
    # results_mae_mean.append(scores['test_neg_mean_absolute_error'].mean())
    # results_r2_mean.append(scores['test_r2'].mean())
    # model_names.append(model_name)
    
    
    #using cross_val_score
    # kfold = KFold(n_splits=10, random_state=7)
    # cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    # results.append(cv_results)
    # means.append(cv_results.mean())
    # stds.append(cv_results.std())
    # model_names.append(model_name)
    # msg = '%s: %f (%f)' % (model_name, cv_results.mean(), cv_results.std())
    # print(msg)
    
    
    # #train/test split
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # MSE = mean_squared_error(y_test,y_pred)
    # print('MSE is:', MSE)
    # print('RMSE is:', math.sqrt(MSE))
    # r2 = r2_score(y_test,y_pred)
    # print('r2 is:', r2)
    # ttr2.append(r2)
    # MAE = mean_absolute_error(y_test,y_pred)
    # print('MAE is:', MAE)
    # ttmse.append(MSE)
    # ttrmse.append(math.sqrt(MSE))
    
    #using shuffle split
    scores = cross_validate(model, X, Y, cv=cv, scoring=scoring)
    results_mse.append(scores['test_neg_mean_squared_error'])
    results_rmse.append(scores['test_neg_root_mean_squared_error'])
    results_mae.append(scores['test_neg_mean_absolute_error'])
    results_r2.append(scores['test_r2'])
    
    results_mse_mean.append(scores['test_neg_mean_squared_error'].mean())
    results_rmse_mean.append(scores['test_neg_root_mean_squared_error'].mean())
    results_mae_mean.append(scores['test_neg_mean_absolute_error'].mean())
    results_r2_mean.append(scores['test_r2'].mean())
    model_names.append(model_name)
    



    
#boxplot algorithm comparison mse
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison MSE')
ax = fig.add_subplot(111)
pyplot.boxplot(results_mse)
ax.set_xticklabels(model_names)


#boxplot algorithm comparison rmse
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison RMSE')
ax = fig.add_subplot(111)
pyplot.boxplot(results_rmse)
ax.set_xticklabels(model_names)


#boxplot algorithm comparison mae
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison MAE')
ax = fig.add_subplot(111)
pyplot.boxplot(results_mae)
ax.set_xticklabels(model_names)


#boxplot algorithm comparison r2
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison r2')
ax = fig.add_subplot(111)
pyplot.boxplot(results_r2)
ax.set_xticklabels(model_names)




#boxplot algorithm comparison
# fig = pyplot.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.plot(means)
# ax.set_xticklabels(model_names)



###COMPARE mean metric values###
# df = pd.DataFrame({'mean r2': results_r2_mean, 'mean rmse': results_rmse_mean, 'mean mse': results_mse_mean, 'mean mae': results_mae_mean }, index=model_names)
# axes = df.plot.line(rot=0, subplots=False, markevery=1, marker='o', markerfacecolor='r')
# #axes.axhline(y=99.918, color='r', label='Benchmark DR 99.918%', ls='--', lw=2)
# axes.legend(loc=4)
# #axes.set_xticks([0,1,2,3,4,5,6,7,8,9, 10])
# axes.set_xticklabels(model_names)
# # for x, y in enumerate(test_dr):
# #     axes.text(x, y, y, ha="left")
# # for x, y in enumerate(train_dr):
# #     axes.text(x, y, y, ha="left") 
# axes.set_xlabel('Classifier')
# axes.set_ylabel('metric values')
# axes.set_title('Comparison of r2 value for selected classifiers')



#Comparison of several metrics
fig = pyplot.figure()
ax1 = fig.add_subplot(411)
df = pd.DataFrame({'mean r2': results_r2_mean}, index=model_names)
ax1.plot(df, markevery=1, marker='o', markerfacecolor='r', label ='mean r2')
ax1.label_outer() #only use outermost label...
ax1.legend(loc='upper left')
ax1.legend(loc=4)
ax1.set_ylabel('R2')
ax1.set_title('Comparison of RMSE, MAE and R2 for selected classifiers')


ax2 = fig.add_subplot(412, sharex=ax1)
df = pd.DataFrame({'mean mae': results_mae_mean})
ax2.plot(df, markevery=1, marker='o', markerfacecolor='r', label ='mean mae')
ax2.label_outer()
ax2.legend(loc='upper left')
ax2.legend(loc=4)
ax2.set_ylabel('MAE')

ax3 = fig.add_subplot(413, sharex=ax1)
df = pd.DataFrame({'mean rmse': results_rmse_mean})
ax3.plot(df, markevery=1, marker='o', markerfacecolor='r', label ='mean rmse')
ax3.label_outer()
ax3.legend(loc='upper left')
ax3.legend(loc=4)
#ax3.set_xticks([0,1,2,3,4,5,6,7,8,9, 10])
#ax3.set_xticklabels(model_names)
ax3.set_ylabel('RMSE')


ax4 = fig.add_subplot(414, sharex=ax1)
df = pd.DataFrame({'mean mse': results_mse_mean})
ax4.plot(df, markevery=1, marker='o', markerfacecolor='r', label ='mean mse')
ax4.legend(loc='upper left')
ax4.legend(loc=4)
#ax3.set_xticks([0,1,2,3,4,5,6,7,8,9, 10])
#ax3.set_xticklabels(model_names)
ax4.set_ylabel('MSE')



#save output
set_option('display.width', 100)
set_option('precision', 3)
saved_result = pd.DataFrame({'mean r2': results_r2_mean,'mean mae': results_mae_mean, 'mean rmse': results_rmse_mean, 'mean mse': results_mse_mean}, index=model_names)
saved_result
saved_result.to_csv("compare_ml_saved_result.csv")
#pyplot.plot(Y)
#plt.plot(model.predict(X_train[:600,]))
#pyplot.show()

