# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:53:42 2020

@author: k_sha
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

filein ="measured.csv"

dfMeasured = read_csv(filein)

releasedatain = "releasedata.csv"

dfReleasedata = read_csv(releasedatain)


#crosswinddatain = "crosswind.csv"

#dfcrosswinddatain = read_csv(crosswinddatain)

#merge two dataframes

merged_Frame = pd.merge(dfMeasured,dfReleasedata, on = "SerieNum" , how="left")

#merged_Frame = pd.merge(merged_Frame,dfcrosswinddatain, on = "SerieNum" , how="left")


#count nans in the dat set
merged_Frame.isnull().sum()

#check columns names
merged_Frame.columns
merged_Frame.shape

#create new column to identify stable/unstable experiments
#Stable indicators:
    #1 Stability Class = F
    #2 Windspeed <=2 m/s "sta_wind"
    #3 h (mixing layer height) / l (Monin Obukhov length ) >10 "sta_met"
    #My criteria will use a composite of 2 and 3, with a check on 1 for verification

#need to add a summary of the range of each variable to my report!!!

merged_Frame['sta_wind'] =  merged_Frame['U_1m  (m/s)'] <= 2.04 #added 2% to prevent a sharp cutoff for values just over 2m/s 
merged_Frame['sta_met'] =  (merged_Frame['Zmix (m)'] / merged_Frame['L (m)']) > 10
merged_Frame['stable'] = (merged_Frame['sta_wind'] & merged_Frame['sta_met']) == True
merged_Frame['stable'] = merged_Frame['stable'].astype(int)
merged_Frame = merged_Frame.drop('sta_wind', 1)
merged_Frame = merged_Frame.drop('sta_met', 1)

#drop nan values
merged_Frame = merged_Frame.dropna()
merged_Frame.shape
merged_Frame.shape



#check columns names
merged_Frame.columns

#check data types
merged_Frame.dtypes

#change material type from object to categorical 
merged_Frame['SerieNum'] = pd.Categorical(merged_Frame['SerieNum'] )

#Extract time of experiement
#merged_Frame['time (hr_min)'] = merged_Frame['Date_x'].str.split().str[1]

#drop date/time columns for now
merged_Frame = merged_Frame.drop('Date_x', 1)
merged_Frame = merged_Frame.drop('Date_y', 1)
#merged_Frame = merged_Frame.drop('time (hr_min)', 1)

#drop other unwanted/not useful columns
#StdW (m/s), 'Zo (m)'

merged_Frame = merged_Frame.drop('StdW (m/s)', 1)
merged_Frame = merged_Frame.drop('Zo (m)', 1)



#keep Dx, Dy, Q, V, Dir, STA, T, Hs
merged_Frame = merged_Frame.drop('DirTo', 1)
merged_Frame = merged_Frame.drop('WindDir_2m', 1)
#merged_Frame = merged_Frame.drop('Ustar (m/s)', 1)
#merged_Frame = merged_Frame.drop('HeatFlux (W/m2)', 1)
#merged_Frame = merged_Frame.drop('Zmix (m)', 1)
merged_Frame = merged_Frame.drop('StdDD (deg)', 1)
#merged_Frame = merged_Frame.drop('Wstar (m/s)', 1)
#Drop calculated crosswind concentrations
# merged_Frame = merged_Frame.drop('Measured 50m', 1)
# merged_Frame = merged_Frame.drop('Measured 100m', 1)
# merged_Frame = merged_Frame.drop('Measured 200m', 1)
# merged_Frame = merged_Frame.drop('Measured 400m', 1)
# merged_Frame = merged_Frame.drop('Measured 800m', 1)


#check columns names
merged_Frame.columns
#Index(['SerieNum', 'Dir', 'RecNum', 'Dist (m)', 'Conc_MG', 'Q (g/s)', 'Hs (m)', 'U_1m  (m/s)',
#       'T (C)', 'L (m)', 'StabilityClass'],
#      dtype='object')



#Transform stability class to categorical and onehot encode
#change material type from object to categorical 
merged_Frame['StabilityClass'] = pd.Categorical(merged_Frame['StabilityClass'])
#le = LabelEncoder()
#merged_Frame['StabilityClass'] = le.fit_transform(merged_Frame['StabilityClass'])

#check types
merged_Frame.dtypes

sc_df = pd.get_dummies(merged_Frame['StabilityClass'], prefix = 'category')
sc_df.shape
sc_df.columns

sc_df.head(5)

#merge both dataframes...
merged_Frame = pd.concat([merged_Frame, sc_df], axis=1)

#drop the "StabilityClass" column
#merged_Frame = merged_Frame.drop('StabilityClass', 1)


#do some summay stats
from pandas import set_option
set_option('display.width', 100)
set_option('precision', 3)
merged_Frame['HeatFlux (W/m2)'].describe()

merged_Frame.describe()



###################################################################


pg_ml_data = merged_Frame

pg_ml_data.shape
pg_ml_data.columns

#remove some experiments -	3, 4, 13 and 14 wind less than 1m/s Armani 2014
# pg_ml_data = pg_ml_data.loc[pg_ml_data['SerieNum'] != 3]
# pg_ml_data = pg_ml_data.loc[pg_ml_data['SerieNum'] != 4]
# pg_ml_data = pg_ml_data.loc[pg_ml_data['SerieNum'] != 13]
# pg_ml_data = pg_ml_data.loc[pg_ml_data['SerieNum'] != 14]
# pg_ml_data.shape

# #armani also says the following:
# #unstable (L < 0) and stable (L > 0) conditions.

# #Remove some more (3, 4, 13, 14, 35, 38, 40, 41, 53, 58)  Marie et al

# pg_ml_data = pg_ml_data.loc[pg_ml_data['SerieNum'] != 35]
# pg_ml_data = pg_ml_data.loc[pg_ml_data['SerieNum'] != 38]
# pg_ml_data = pg_ml_data.loc[pg_ml_data['SerieNum'] != 40]
# pg_ml_data = pg_ml_data.loc[pg_ml_data['SerieNum'] != 41]
# pg_ml_data = pg_ml_data.loc[pg_ml_data['SerieNum'] != 53]
# pg_ml_data = pg_ml_data.loc[pg_ml_data['SerieNum'] != 58]
# pg_ml_data.shape




#add X,Y columns

column0 = pg_ml_data['RecNum']  # angle /Rec Num
column1 = pg_ml_data['Dist (m)']  # radial distance / Dist (m)


distX = column1 * np.cos(np.deg2rad(column0))
distY = column1 * np.sin(np.deg2rad(column0))


pg_ml_data['X'] = distX
pg_ml_data['Y'] = distY


#plot to visualise X/Y coordinates
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(distX,distY, label = "Sensor location")
plt.xlabel("Downwind distance, X (m)")
plt.ylabel("Crosswind distance, Y (m)")
plt.legend(loc='upper right')
plt.show()

#remove dodgy data
#remove any negative concentration values
pg_ml_data = pg_ml_data.loc[pg_ml_data['Conc_MG'] > 0]
pg_ml_data.shape

len(pg_ml_data.columns)



#Change type of first 4 features to categorical
#pg_ml_data['SerieNum'] = pd.Categorical(pg_ml_data['SerieNum'] )
#pg_ml_data['Dir'] = pd.Categorical(pg_ml_data['Dir'] )
#pg_ml_data['RecNum'] = pd.Categorical(pg_ml_data['RecNum'] )
#pg_ml_data['Dist (m)'] = pd.Categorical(pg_ml_data['Dist (m)'] )


#drop 'SerieNum', 'Dir', 'RecNum', 'Dist (m)', 'StabilityClass'
#reaarange wuth target at the end
#we are at 19 features here
pg_ml_data = pg_ml_data[['Q (g/s)', 'Hs (m)', 'U_1m  (m/s)',
       'Ustar (m/s)', 'HeatFlux (W/m2)', 'Zmix (m)', 'T (C)', 'Wstar (m/s)', 'L (m)',
       'stable', 'category_A', 'category_B', 'category_C', 'category_D',
       'category_E', 'category_F', 'X', 'Y','Conc_MG']]




# #Breakdown by Arc - 50m
# pg_ml_data_50 = pg_ml_data.loc[pg_ml_data['Dist (m)'] == 50]
# pg_ml_data_50.shape
# arc1 = "pg_ml_data_50.csv"
# pg_ml_data_50.to_csv(arc1, index=False)



# #Breakdown by Arc - 100m
# pg_ml_data_100 = pg_ml_data.loc[pg_ml_data['Dist (m)'] == 100]
# pg_ml_data_100.shape
# arc2 = "pg_ml_data_100.csv"
# pg_ml_data_100.to_csv(arc2, index=False)


# #Breakdown by Arc - 200m
# pg_ml_data_200 = pg_ml_data.loc[pg_ml_data['Dist (m)'] == 200]
# pg_ml_data_200.shape
# arc3 = "pg_ml_data_200.csv"
# pg_ml_data_200.to_csv(arc3, index=False)

# #Breakdown by Arc - 400m
# pg_ml_data_400 = pg_ml_data.loc[pg_ml_data['Dist (m)'] == 400]
# pg_ml_data_400.shape
# arc4 = "pg_ml_data_400.csv"
# pg_ml_data_400.to_csv(arc4)


# #Breakdown by Arc - 800m
# pg_ml_data_800 = pg_ml_data.loc[pg_ml_data['Dist (m)'] == 800]
# pg_ml_data_800.shape
# arc5 = "pg_ml_data_800.csv"
# pg_ml_data_800.to_csv(arc5, index=False)




#feature plots
#9 features in total
fig = plt.figure()
ax = fig.add_subplot(9,1,1)
ax.plot(pg_ml_data['Q (g/s)'], label ='Q (g/s)')
ax.legend(loc='upper left')

ax = fig.add_subplot(9,1,2)
ax.plot(pg_ml_data['Hs (m)'], label ='Hs (m)')
ax.legend(loc='upper left')
ax = fig.add_subplot(9,1,3)
ax.plot(pg_ml_data['U_1m  (m/s)'], label ='U_1m  (m/s)')
ax.legend(loc='upper left')
ax = fig.add_subplot(9,1,4)
ax.plot(pg_ml_data['Ustar (m/s)'], label ='Ustar (m/s)')
ax.legend(loc='upper left')
ax = fig.add_subplot(9,1,5)
ax.plot(np.abs(pg_ml_data['HeatFlux (W/m2)']), label ='HeatFlux (W/m2)')
ax.legend(loc='upper left')
ax = fig.add_subplot(9,1,6)
ax.plot(pg_ml_data['Zmix (m)'], label ='Zmix (m)')
ax.legend(loc='upper left')
ax = fig.add_subplot(9,1,7)
ax.plot(pg_ml_data['T (C)'], label ='T (C)')
ax.legend(loc='upper left')
ax = fig.add_subplot(9,1,8)
ax.plot(pg_ml_data['Wstar (m/s)'], label ='Wstar (m/s)')
ax.legend(loc='upper left')
ax = fig.add_subplot(9,1,9)
ax.plot(pg_ml_data['L (m)'], label ='L (m)')
ax.legend(loc='upper left')
plt.show()



pg_ml_data.columns



#plots for reporting
fig3D = plt.figure()
fig3D.subplots_adjust(hspace=1, wspace=1)
ax3D = fig3D.add_subplot(1,1,1, projection = '3d')
ax3D.scatter(pg_ml_data['X'],pg_ml_data['Y'], pg_ml_data['Conc_MG'], label="Conc_MG")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend(loc='upper right')
#ax3D.plot3D(pg_ml_data['X'],pg_ml_data['Y'], pg_ml_data['conc'])



#correlation matrix plot

pg_ml_data_corr = pg_ml_data[['Q (g/s)', 'Hs (m)', 'U_1m  (m/s)',
        'Ustar (m/s)', 'HeatFlux (W/m2)', 'Zmix (m)', 'T (C)', 'Wstar (m/s)', 'L (m)', 'Conc_MG']]
correlations = pg_ml_data.corr()
names = ['Q (g/s)', 'Hs (m)', 'U_1m  (m/s)',
       'Ustar (m/s)', 'HeatFlux (W/m2)', 'Zmix (m)', 'T (C)', 'Wstar (m/s)', 'L (m)', 'Conc_MG']
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation=90)
ax.set_yticklabels(names)
plt.show()



#scatter plot
pd.plotting.scatter_matrix(pg_ml_data_corr, figsize=[20,20])
plt.show()


#box plots
pg_ml_data_corr.plot(kind='box', subplots=True, layout=(2,5), sharex=False, sharey=False)
plt.tight_layout()
plt.show()

#focussing on the target variable
#box plot
pg_ml_data['Conc_MG'].plot(kind='box')
plt.show()

#hist plot
pg_ml_data['Conc_MG'].plot(kind='hist')
plt.show()


#corrected hist plot
target = pg_ml_data['Conc_MG']
target_log_xform = np.log(target)
target_log_xform.plot(kind='hist')
plt.show()


#corrected hist plot 2
target2 = pg_ml_data['Conc_MG']
target2_log_xform = np.log1p(target2)
target2_log_xform.plot(kind='hist')
plt.show()



#corrected hist plot 3
target3 = pg_ml_data['Conc_MG']
target3_log_xform = np.log10(target3)
target3_log_xform.plot(kind='hist')
plt.show()

#corrected hist plot 4
target4 = pg_ml_data['Conc_MG']
target4_log_xform = np.log2(target4)
target4_log_xform.plot(kind='hist')
plt.show()

#corrected hist plot 5
target5 = pg_ml_data['Conc_MG']*1000
target5_log_xform = np.log10(target5)
target5_log_xform.plot(kind='hist')
plt.show()


target5_log_xform.plot(kind='box')
plt.show()


#descriptive analytics
#first 5 features
pd.set_option('display.width',0)
pd.set_option('precision',2)
description = pg_ml_data[['Q (g/s)', 'Hs (m)', 'U_1m  (m/s)','Ustar (m/s)', 'HeatFlux (W/m2)']].describe()
print(description)



#next 5 features
pd.set_option('display.width',0)
pd.set_option('precision',2)
description = pg_ml_data[['Zmix (m)', 'T (C)', 'Wstar (m/s)', 'L (m)', 'Conc_MG']].describe()
print(description)






#Save for use in other modelling

# Added 08/08/23

#Original concentration units
pg_ml_data_init_conc_mg = pg_ml_data
pg_ml_data_init_conc_mg.shape
#(8173, 23)
init_conc_mg = "pg_ml_data_init_conc_mg.csv"
pg_ml_data_init_conc_mg.to_csv(init_conc_mg, index=False)




#log transform target variable
pg_ml_data['Conc_MG'] = np.log10(pg_ml_data['Conc_MG'] * 1000)


#Create multiple datsets

#All data
pg_ml_data_alldata = pg_ml_data
pg_ml_data_alldata.shape
#(8173, 23)
alldata = "pg_ml_data_alldata.csv"
pg_ml_data_alldata.to_csv(alldata, index=False)


#Stable/Unstable

#unstable dataset
pg_ml_data_unstable = pg_ml_data.loc[pg_ml_data['stable'] == 1]
pg_ml_data_unstable.shape
#(1461, 19)
unstable = "pg_ml_data_unstable.csv"
pg_ml_data_unstable.to_csv(unstable, index=False)

#stable dataset
pg_ml_data_stable = pg_ml_data.loc[pg_ml_data['stable'] == 0]
pg_ml_data_stable.shape
#(6709, 19)
stable = "pg_ml_data_stable.csv"
pg_ml_data_stable.to_csv(stable, index=False)
