#TU Delft
#Biomechatronics-BME
#Lian Wu
#4230558
#---------------------------Logic of this File---------------------------------
# 1. Import data into python and perform basic data analysis
# 2. Restructure names to ensure compatibility
# 3. Remove unrelated variables(features) upon first inspection
# 4. Fill NAN values with 0 if necessary
# 5. Update Recurrence columns for better formatting
# 6. Perform Ordinary Encoding
# 7. Perform One Hot Encoding
# 8. Perform Feature Extranction or Selection
# 9. Apply necessary methods to incoorporate temporal features
# 10. 
#
# *REMEBER TO CHOOSE A PERFORMANCE METRIC (Accuracy!!)
#
#Full example of machine learning
#https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/?utm_source=time-series-forecasting-codes-python

#%%
#Import packages
import pandas as pd
#pip install googletrans
#conda install googletrans
#from translate import Translator
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# from scipy.stats import spearmanr

#import numpy
#import matplotlib.pyplot as plt
#from pandas import read_csv
#import math
#import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error

from pandas import DataFrame
from pandas import concat

#---------------------------Data Analysis------------------------------------------------------------------------------------------

#Import data
df = pd.read_excel ('data_en.xlsx', sheet_name='Sheet1')

#----------------------------Delete useless signs-----------------------------------------------------------------------------------------

def delete_useless_signs(df):
    title_lst = list(df.columns)
    
    for i in range(0,len(title_lst)):
        current_title_lst = title_lst[i]
        title_lst[i] = current_title_lst.replace(" ", "")
        current_title_lst = title_lst[i]
        title_lst[i] = current_title_lst.replace("'", "")
        current_title_lst = title_lst[i]
        title_lst[i] = current_title_lst.replace("?", "")
        #print(i)
        
    df.columns = title_lst
    title_lst = list(df.columns)

#Removing useless variables
    #del caxurrent_title_lst, title_lst , i
    return df

df = delete_useless_signs(df)

#---------------------------Data Analysis------------------------------------------------------------------------------------------

def drop_columns(df):
    #Drop colomns of de-identified columns
    df = df.drop(["Timeofanswer"],axis=1)
    
    #Drop columns with long paragraphs of answers
    df = df.drop(["Canyouexplaininyourownwordswhathappened","OVERUSE-Canyouexplaininyourownwordswhathappened"],axis=1)
    
    df = df.drop(["ACUUT-Whatisthenatureoftheinjury","ACUUT-Whatisthecauseoftheinjury"],axis=1)
    
    df = df.drop(["Howwellhaveyousleptinthepast7days"],axis=1)
    
    df = df.drop(['ILLNESS-WhatdiseasesymptomshaveyouexperiencedinthepastweekIndicatehereallthesymptoms'],axis=1)

    return df

df = drop_columns(df)

#Fill in nan values with zero (0)
df = df.fillna(0)

#update recurrence values for better viewing, double check if the lowest value is 79 or not
df['Recurrence'] = df['Recurrence']+79

#Update title list
title_lst = list(df.columns)

#----------------------Ordinal Encoding----------------------------------------------------------------------------------------------------------------------------------
#Most of this is ordinal encoding. Values that require one hot encoding require to be tranformed to ordinal first, then use another method to one hot encode
#Source: https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159

def ordinalencoding(df):

    title_lst = list(df.columns)
    string_title_lst=title_lst[5:21]
    
    for i in range(len(string_title_lst)):
        #valuecounts = df[string_title_lst[i]].value_counts()
        
        #Haveyouexperiencedaphysicalcomplaintduringthepast7daysduringexercise
        if i == 0:
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I fully participated without complaints", 0) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I fully participated, but was bothered by a physical complaint", 1) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I participated in part because of a physical complaint", 2) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I did not participate at all due to a physical complaint", 3) 
        
        #Towhatextenthaveyouadjustedyourtrainingorparticipationincompetitionsinthepast7daysduetothisphysicalcomplaint
        elif i == 1:
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I have not adjusted this", 0) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("Somewhat", 1) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("A lot of", 3) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("Mediocre", 2) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I could not participate at all", 4) 
        
        #Towhatextenthaveyounoticedinthepast7daysthatthisphysicalcomplainthasaffectedyourperformance
        elif i == 2:
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("My performance was not affected", 0) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("Somewhat", 1) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("A lot of", 3) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("Mediocre", 2) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I could not participate at all", 4) 
        
        #Towhatextentdidyouexperiencethesymptomsofthisphysicalcomplaintinthepast7days
        elif i == 3:
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I wasn't bothered", 0) #Ik had geen last
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("Somewhat", 1) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("A lot of", 3) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("Mediocre", 2) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I could not participate at all", 4) #Ik heb geheel niet kunnen deelnemen
        
        #Howmanydaysinthelastweekhaveyoubeenunabletoparticipatefullyorcompletelyinatrainingorcompetitionduetothisphysicalcomplaint
        elif i == 4:
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("Fully participated", 0) #Volledig deelgenomen
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("All", 5) #Alle 
    
        #ACUUT-Isthisthefirsttimeyouhavereportedthisphysicalcomplaint
        elif i == 6:
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("Yes", 0) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("No, I also reported the same problem last time", 1)
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("No, I reported the same problem earlier, but that was longer ago", 1)
        
        #OVERUSE-Isthisthefirsttimeyouhavereportedthisphysicalcomplaint
        elif i == 9:
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("Yes", 0) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("No, I also reported the same problem last time", 1)
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("No, I reported the same problem earlier, but that was longer ago", 1)
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("Damn it", 1)
            
        #ILLNESS-Isthisthefirsttimeyouhavereportedthisphysicalcomplaint
        elif i == 12:
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("Yes", 0) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("No, I also reported the same problem last time", 1)
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("No, I reported the same problem earlier, but that was longer ago", 1) 

        #Scale the 0-100 values to 0-1:
        #Howresteddoyoucurrentlyfeel
        elif i == 13:
            df[string_title_lst[i]]= df[string_title_lst[i]]/100
     
        #Howenergeticdoyoufeelatthemoment
        elif i == 14:
            df[string_title_lst[i]]= df[string_title_lst[i]]/100
        
        #Howresteddoyoucurrentlyfeel
        elif i == 15:
            df[string_title_lst[i]]= df[string_title_lst[i]]/100

    return df

df = ordinalencoding(df)

#---------------------------------Onehot encoding-----------------------------------------------------------------------------

def onehotencoding(df):

    title_lst = list(df.columns)
    string_title_lst=title_lst[5:21]
    
    obj_columns = []
    for i in range(0,5):
        number =[5,7,8,10,11]
        obj_columns.append(string_title_lst[number[i]])
    
    newdf = df[['Doesthephysicalcomplaintthatyouhaveexperiencedpainordiscomfortinthepast7daysconcernaninjuryorillness','ACUUT-Duringwhichactivitydidtheinjuryoccur','ACUUT-Whereistheinjury','OVERUSE-Duringwhichactivitydidtheinjuryoccur','OVERUSE-Whereistheinjury']]
    
    df_encoded = pd.get_dummies(data=newdf, columns=obj_columns)
    
    #drop the columns that were just onehot encoded
    
    for i in range(0,len(obj_columns)):
        df = df.drop([obj_columns[i]],axis=1)
        
    # merge with main df bridge_df on key values
    df = df.join(df_encoded)

    return df, df_encoded

df,df_encoded = onehotencoding(df)

#--------------------------Determine Injury Status by 4 Question Indicators-------------------------------------------------
#Logic of injury status definition
    
#Columns for injury status indication
#Participations - Haveyouexperiencedaphysicalcomplaintduringthepast7daysduringexercise
#Modified training - Towhatextenthaveyouadjustyourtrainingorparticipationincompetitionsinthepast7daysduetothisphysicalcomplaint
#Performance - Towhatextenthaveyounoticedinthepast7daysthatthisphysicalcomplainthasaffectedyourperformance
#Symptoms - Towhatextentdidyouexperiencethesymptomsofthisphysicalcomplaintinthepast7days

#n = number of weeks to be taken into account to determine injury status

def injurystatusdef(df):
        
    #Define injury weight
    injury_weight = [0.25 , 0.25 , 0.25 , 0.25]
    final_weight_0 = [0.5 , 0.3 , 0.15 , 0.5]
    final_weight_1 = [0.6 , 0.3 , 0.1]
    final_weight_2 = [0.7 , 0.3]
    final_weight_3 = [1]
    
    #Data management
    
    df = df.sort_values(by=['Participantsname', 'Recurrence'])
    
    df_injury = df[["Participantsname","Recurrence","Haveyouexperiencedaphysicalcomplaintduringthepast7daysduringexercise","Towhatextenthaveyouadjustedyourtrainingorparticipationincompetitionsinthepast7daysduetothisphysicalcomplaint","Towhatextenthaveyounoticedinthepast7daysthatthisphysicalcomplainthasaffectedyourperformance","Towhatextentdidyouexperiencethesymptomsofthisphysicalcomplaintinthepast7days"]]
    df_injury["Haveyouexperiencedaphysicalcomplaintduringthepast7daysduringexercise"] = df_injury["Haveyouexperiencedaphysicalcomplaintduringthepast7daysduringexercise"]*injury_weight[0]
    df_injury["Towhatextenthaveyouadjustedyourtrainingorparticipationincompetitionsinthepast7daysduetothisphysicalcomplaint"] = df_injury["Towhatextenthaveyouadjustedyourtrainingorparticipationincompetitionsinthepast7daysduetothisphysicalcomplaint"]*injury_weight[1]
    df_injury["Towhatextenthaveyounoticedinthepast7daysthatthisphysicalcomplainthasaffectedyourperformance"] = df_injury["Towhatextenthaveyounoticedinthepast7daysthatthisphysicalcomplainthasaffectedyourperformance"]*injury_weight[2]
    df_injury["Towhatextentdidyouexperiencethesymptomsofthisphysicalcomplaintinthepast7days"] = df_injury["Towhatextentdidyouexperiencethesymptomsofthisphysicalcomplaintinthepast7days"]*injury_weight[3]
    df["Firstscore"] = df_injury["Haveyouexperiencedaphysicalcomplaintduringthepast7daysduringexercise"] + df_injury["Towhatextenthaveyouadjustedyourtrainingorparticipationincompetitionsinthepast7daysduetothisphysicalcomplaint"] + df_injury["Towhatextenthaveyounoticedinthepast7daysthatthisphysicalcomplainthasaffectedyourperformance"] + df_injury["Towhatextentdidyouexperiencethesymptomsofthisphysicalcomplaintinthepast7days"]
    df_injury["Firstscore"] = df[["Firstscore"]]
    participantsname = df_injury[["Participantsname"]].to_numpy()
    firstscore = df_injury[["Firstscore"]].to_numpy()
    
    final_injury = np.zeros(len(df_injury))

    for i in range(3,len(df_injury)):
        j = i - 1
        k = i - 2
        l = i - 3
        
        if participantsname[i] == participantsname[j] == participantsname[k] == participantsname[l]:    
            if firstscore[j] == firstscore[k] == firstscore[l] == 0:
                final_injury[i] = 1 * firstscore[i]
            if (firstscore[k] == firstscore[l] == 0) and firstscore[j] != 0:
                final_injury[i] = 0.6 * firstscore[i] + 0.4 * firstscore[j]    
            if firstscore[l] == 0 and firstscore[j] != 0 and firstscore[k] !=0:
                final_injury[i] = 0.5 * firstscore[i] + 0.25 * firstscore[j] + 0.15 * firstscore[k]
            else:
                final_injury[i] = final_weight_0[0] * firstscore[i] + final_weight_0[1] * firstscore[j] + final_weight_0[2] * firstscore[k] + final_weight_0[3] * firstscore[l]
    
        if participantsname[i] == participantsname[j] == participantsname[k] != participantsname[l]:      
            final_injury[i] = final_weight_1[0] * firstscore[i] + final_weight_1[1] * firstscore[j] + final_weight_1[2] * firstscore[k]
            final_injury[j] = final_weight_2[0] * firstscore[j] + final_weight_2[1] * firstscore[k]
            final_injury[k] = final_weight_3[0] * firstscore[k]
    
    df["Finalinjury"] = final_injury
    df["Finalinjury"] = np.ceil(df["Finalinjury"]).astype(int)
    df = df.drop("Firstscore", axis=1)
    return df

df = injurystatusdef(df)

#---------------------------------Sliding Window-----------------------------------------
#http://ethen8181.github.io/machine-learning/time_series/3_supervised_time_series.html
#https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/


#This works!!!!
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
        (this means that dataframe needs to be modified into a list or array)
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df1 = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df1.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df1.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#print(data_sliding)
# Number 4 is for the amount of lag you want, 4 means 4 lags beind
#I HAVE TO MAKE SURE THAT THE LAST COLUMN IS WHAT WE ARE TRYING TO PREDICT
#variable data is the dataset that can be used for ML input
#https://datatofish.com/export-dataframe-to-csv/

#---------------------------------Forward Validation-----------------------------------------
#https://alphascientist.com/walk_forward_model_building.html

#Pipeline
#slice out only the data which has happened up until that point
#train a model with that data
#save a copy of that fitted model object in a pandas Series called models which is indexed by the recalculation date.

#Rewriting my own function for walking forward

def walking_forward(data, n, personalised , athelete_name):
    #n = the number of entries wanted to be included in the data, do each the length of data for each athlete before entring number
    #personalised = 1 if input required is for personalised model
    
    if personalised == 1:
        dict_of_athletes = dict(tuple(data.groupby("Participantsname")))
        data_walking_X = dict()
        data_walking_Y = dict()
        
        athlete_data = dict_of_athletes[athlete_name]
        
        athlete_data = athlete_data.drop(["Recurrence","Participantsname"], axis=1)
        
        athlete_data_X = athlete_data.drop("Finalinjury", axis=1)
        athlete_data_Y = athlete_data("Finalinjury")
        
        for i in range(1,1600):
            data_per_X = athlete_data_X.head(i)
            data_per_Y = athlete_data_Y.head(i)
            data_walking_X[i] = data_per_X
            data_walking_Y[i] = data_per_Y
        
        test_X = athlete_data_X.tail(500)
        test_Y = athlete_data_Y.tail(500)
        
    if personalised == 0:
        #the data in recurrence and then athlete
        data_walking_X = dict()
        data_walking_Y = dict()
        
        athlete_data = data.sort_values(by=['Recurrence','Participantsname'])
        athlete_data = athlete_data.drop(["Recurrence","Participantsname"],axis=1)
        
        athlete_data_X = athlete_data.drop("Finalinjury", axis=1)
        athlete_data_Y = athlete_data["Finalinjury"]
        
        #slice the data base on the number of entries
        j = 0
        for i in range(1,1600,300):
            data_per_X = athlete_data_X.head(i)
            data_per_Y = athlete_data_Y.head(i)
            data_walking_X[j] = data_per_X
            data_walking_Y[j] = data_per_Y
            j = j + 1
            
        test_X = athlete_data_X.tail(500)
        test_Y = athlete_data_Y.tail(500)
        
    return data_walking_X, data_walking_Y, test_X, test_Y

#%%#-----------------------Random Forest data sliding----------------------------------------------

#https://blog.goodaudience.com/introduction-to-random-forest-algorithm-with-python-9efd1d8f0157
#https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix

#Normal data
# y = df['Finalinjury']
# X = df.drop(['Finalinjury','Participantsname'], axis = 1)

#Sliding window
test_df = df
test_df = test_df.drop("Participantsname",axis=1) #athlete name
test_df = test_df.drop("Recurrence",axis=1) #recurrence
values = test_df
data_sliding = series_to_supervised(values, 4)
y = data_sliding['var71(t)']
X = data_sliding.drop(['var71(t)'], axis = 1)

# Split the dataset to trainand test data (Sliding Window)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110, 120],
    'max_features': [8,9,10,15,20,40,60,80,100],
    'min_samples_leaf': [5, 6, 7, 8, 9, 10],
    'min_samples_split': [8, 10, 12, 14, 16],
    'n_estimators': [100, 200, 300, 400, 500]
}

RF_model = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = RF_model, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

import time
start_time = time.time()

grid_result = grid_search.fit(train_X, train_y)

elapsed_time = time.time() - start_time
print("This simulation ran for",elapsed_time, "seconds, around",elapsed_time/60,"minutes.")
    
best_grid = grid_search.best_estimator_

#features_used_RF = RF_model.feature_importances_

# #------------------------------------
# def evaluate(model, test_X, test_y):
#     predictions = model.predict(test_X)
#     errors = abs(predictions - test_y)
#     mape = 100 * np.mean(errors / test_y)
#     accuracy = 100 - mape
#     print('Model Performance')
#     print('Accuracy = {:0.2f}%.'.format(accuracy))
    
#     return accuracy
# grid_accuracy = evaluate(best_grid, test_X, test_y)

# RF_predictions = predict(best_grid,test_X)
# score = accuracy_score(test_y ,RF_predictions)
# print("Accuracy is", score*100 ,"%")

colum_names = ["best_params","best_score" ]
best = pd.DataFrame(columns=colum_names)
best_params = []
best_score = []

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    best_params.append(param)
    best_score.append(mean)
    print("%f (%f) with: %r" % (mean, stdev, param))
    
best["best_params"] = best_params
best["best_score"] = best_score

best = best.sort_values(by=["best_score"],ascending=False)
#Accuracy without sliding window: 0.7793
#Accuracy with sliding window: 0.86835

best.to_csv(r'C:\Users\Lian\OneDrive\Phase 4(With new data) - 27.03.2020\Script\Parameters\'RD_data_sliding.csv')

#Sliding window actualy improved the accuracy of the model, next try to impliment forward method and see.

#%%#-----------------------Random Forest data sliding-specialised model----------------------------------------------
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix

#Sliding window
test_df = df
test_df = test_df.drop("Participantsname",axis=1) #athlete name
test_df = test_df.drop("Recurrence",axis=1) #recurrence
values = test_df
data_sliding = series_to_supervised(values, 4)
y = data_sliding['var71(t)']
X = data_sliding.drop(['var71(t)'], axis = 1)

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2,3,4,5])
n_classes = y.shape[1]

import time
start_time = time.time()

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)

parameters = {'bootstrap': True,
              'min_samples_leaf': 5,
              'n_estimators': 100, 
              'min_samples_split': 8,
              'max_features': 100,
              'max_depth': 80,
              'max_leaf_nodes': None}

RF_model = RandomForestClassifier(**parameters)

RF_model.fit(train_X, train_y)

RF_predictions = RF_model.predict(test_X)
score = accuracy_score(test_y ,RF_predictions)
print("Accuracy is", score*100 ,"%")

elapsed_time = time.time() - start_time

colum_name = ["test","predicted" ]
compare = pd.DataFrame(columns=colum_name)

compare["test"] = test_y
compare["predicted"] = RF_predictions


compare1 = compare[compare["test"]==1]
print("predicted to be 1, the other values are: ")
print(compare1["predicted"].value_counts())

compare2 = compare[compare["test"]==2]
print("predicted to be 2, the other values are: ")
print(compare2["predicted"].value_counts())


compare3 = compare[compare["test"]==3]
print("predicted to be 3, the other values are: ")
print(compare3["predicted"].value_counts())

compare4 = compare[compare["test"]==4]
print("predicted to be 4, the other values are: ")
print(compare4["predicted"].value_counts())

compare5 = compare[compare["test"]==5]
print("predicted to be 5, the other values are: ")
print(compare5["predicted"].value_counts())


features_used_RF = RF_model.feature_importances_

x_axis = np.arange(len(RF_predictions))

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
plt.figure(figsize=(9,3))
plt.plot(x_axis, test_y,'o',markersize=2,markeredgecolor='gray')
plt.plot(x_axis, RF_predictions,'x',markersize=1)
plt.title('Model Prediction VS Actual Values')
plt.ylabel('Class')
plt.xlabel('Time Step')
plt.legend(['Actual','Predicted'], loc='upper left')
plt.rcParams["figure.figsize"] = [16,9]
plt.show()


x_feature = np.arange(len(features_used_RF))
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
plt.figure(figsize=(9,3))
plt.plot(x_feature, features_used_RF)
plt.title('Feature Importance of RF using Data with SW')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.rcParams["figure.figsize"] = [16,9]
plt.show()


#ROC curve

from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

lw = 6
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y[:, i], RF_predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), RF_predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','green','yellow'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


#%%#------------------------------Forward Walking---------------------------------
#https://blog.goodaudience.com/introduction-to-random-forest-algorithm-with-python-9efd1d8f0157
#https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix

#Normal data
# y = df['Finalinjury']
# X = df.drop(['Finalinjury','Participantsname'], axis = 1)

#Sliding window
# values = test_df
# data_sliding = series_to_supervised(values, 4)
# y = data_sliding['var71(t)']
# X = data_sliding.drop(['var71(t)'], axis = 1)

# Split the dataset to trainand test data (Sliding Window)
#train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)

#Walking Forward
X,y,test_X,test_Y = walking_forward(df,2041,0,0.0)

Predicted = []
Actual = []

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110, 120],
    'max_features': [8,9,10,15,20,40,60,80,100],
    'min_samples_leaf': [5, 6, 7, 8, 9, 10],
    'min_samples_split': [8, 10, 12, 14, 16],
    'n_estimators': [100, 200, 300, 400, 500]
}

RF_model = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = RF_model, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
colum_names = ["best_params","best_score" ]
best = pd.DataFrame(columns=colum_names)
best_params = []
best_score = []

import time
start_time = time.time()

for i in range(1,len(X)):
    train_X = X[i]
    train_y = y[i]
    
    # test_X1 = X[i+1]
    # test_y1 = y[i+1]
    
    # test_X = test_X1.iloc[-1:]
    # test_y = test_y1.iloc[-1:]
    
    # Fit the grid search to the data
    grid_result = grid_search.fit(train_X, train_y)
    
    best_grid = grid_search.best_estimator_
    
    best_params.append(grid_result.best_params_)
    best_score.append(grid_result.best_score_)
    
    #RF_predictions = predict(best_grid, test_X)
    #Predicted.append(RF_predictions)
    #Actual.append(test_y.values.tolist())

    #score = accuracy_score(test_y ,RF_predictions)
    print (best_grid)
elapsed_time = time.time() - start_time
print("This simulation ran for",elapsed_time, "seconds, around",elapsed_time/60,"minutes.")

best["best_params"] = best_params
best["best_score"] = best_score

best = best.sort_values(by=["best_score"],ascending=False)

best.to_csv(r'C:\Users\Lian\OneDrive\Phase 4(With new data) - 27.03.2020\Script\Parameters\RD_forward_chaining.csv', index = True)

# Actual = pd.Series(Actual).astype(str)
# Predicted = pd.Series(Predicted).astype(str)

# score = accuracy_score(Actual ,Predicted)
# print("Accuracy is", score*100 ,"%")

# plt.plot(history.history['acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

#features_used_RF = RF_model.feature_importances_

#Accuracy without sliding window: 0.7793
#Accuracy with sliding window: 0.86835

#Sliding window actualy improved the accuracy of the model, next try to impliment forward method and see.

#%%#------------------------------Forward Walking-specialised model---------------------------------

import time
start_time = time.time()

X,y,test_X,test_Y = walking_forward(df,2041,0,0.0)

parameters = {'bootstrap': True,
              'min_samples_leaf': 5,
              'n_estimators': 100, 
              'min_samples_split': 8,
              'max_features': 70,
              'max_depth': 80,
              'max_leaf_nodes': None}

RF_model = RandomForestClassifier(**parameters)

for i in range(1,len(X)):
    train_X = X[i]
    train_y = y[i]
    
    RF_model = RF_model.fit(train_X, train_y)
    

RF_predictions = RF_model.predict(test_X)
score = accuracy_score(test_Y ,RF_predictions)
print("Accuracy is", score*100 ,"%")

elapsed_time = time.time() - start_time

colum_name = ["test","predicted" ]
compare = pd.DataFrame(columns=colum_name)

compare["test"] = test_Y
compare["predicted"] = RF_predictions


compare1 = compare[compare["test"]==1]
print("predicted to be 1, the other values are: ")
print(compare1["predicted"].value_counts())

compare2 = compare[compare["test"]==2]
print("predicted to be 2, the other values are: ")
print(compare2["predicted"].value_counts())


compare3 = compare[compare["test"]==3]
print("predicted to be 3, the other values are: ")
print(compare3["predicted"].value_counts())

compare4 = compare[compare["test"]==4]
print("predicted to be 4, the other values are: ")
print(compare4["predicted"].value_counts())

compare5 = compare[compare["test"]==5]
print("predicted to be 5, the other values are: ")
print(compare5["predicted"].value_counts())


features_used_RF = RF_model.feature_importances_

x_axis = np.arange(len(RF_predictions))

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
plt.figure(figsize=(9,3))
plt.plot(x_axis, test_Y,'o',markersize=2,markeredgecolor='gray')
plt.plot(x_axis, RF_predictions,'x',markersize=1)
plt.title('Model Prediction VS Actual Values')
plt.ylabel('Class')
plt.xlabel('Time Step')
plt.legend(['Actual','Predicted'], loc='upper left')
plt.rcParams["figure.figsize"] = [16,9]
plt.show()


x_feature = np.arange(len(features_used_RF))
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
plt.figure(figsize=(9,3))
plt.plot(x_feature, features_used_RF)
plt.title('Feature Importance of RF using Data with FC')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.rcParams["figure.figsize"] = [16,9]
plt.show()








