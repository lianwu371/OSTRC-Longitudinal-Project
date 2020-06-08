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
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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

# Check classes balance
df.count()

#Fill in nan values with zero (0)
df = df.fillna(0)

# Check for null values values
null_numbers = df.isnull().sum()
del null_numbers

#showing a statistical overview of imported data
stats = df.describe() 

#update recurrence values for better viewing, double check if the lowest value is 79 or not
df['Recurrence'] = df['Recurrence']+79

    #here, there will be a loop for every variable so that each column's value can be encoded properly
    #for example, if "varibale" == "certain string":
                    # value_count all the possible answers
                    # append it to a new list
                    # manually check all the answers and assign a number to each answer
                    # change the strings to their respective number
                    # repeat this for all of the variables, there are 16 of them

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
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I fully participated without complaints", 1) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I fully participated, but was bothered by a physical complaint", 2) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I participated in part because of a physical complaint", 3) 
            df[string_title_lst[i]]= df[string_title_lst[i]].replace("I did not participate at all due to a physical complaint", 4) 
        
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
#Onehot encoding
#filter out only the columns for one hot encoding
#put them into 1 dataframe

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
    
    #Delete non-relevant variables
    #del i,number,obj_columns, df_encoded, newdf

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
    
    injury_list = np.zeros(len(df_injury))
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

test_df = df
test_df = test_df.drop("Participantsname",axis=1) #athlete name
test_df = test_df.drop("Recurrence",axis=1) #recurrence

values = test_df
data_sliding = series_to_supervised(values, 4)

del values, test_df, df_encoded, stats

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

#%%#---------------------------------LSTM-----------------------------------------
#https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/
#
#https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/
#https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/


#https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

#https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

#https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/

#Prepare the proper format for each of the time effect method
#sliding window = seperate each into 4 weeks and feed it into the model moving week by week

#---------------------------------Data Preperation---------------------------------

from sklearn.metrics import accuracy_score

def lstm_sliding(df):
    
    #Remove unnecessary columns
    data_sliding = df.drop(["Participantsname","Recurrence"],axis=1)
    
    Y_values = data_sliding[data_sliding.columns[len(data_sliding.columns)-1]].values
    X_values = data_sliding.drop(data_sliding.columns[-1],axis=1)
    X_values = X_values.values
    
    X = X_values.reshape((X_values.shape[0], 1, X_values.shape[1]))
    Y = Y_values
    
    split_value = 1500;
    test_X = X[split_value:,:]
    test_Y = Y[split_value:]
    
    train_X = X[:split_value,:]
    train_Y = Y[:split_value]
    
    #fit the LSTM network
    
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(5))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adamax',metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=50, batch_size=200, verbose=1)
    print(model.summary())
    
    #LOSS Functions (before seperating test and training)
    #binary_crossentropy (about 75% accuracy, 52.31% with sliding window and new data)
    #mean_squared_error (this has a very high accuracy level 85%, 63.59% with sliding window and new data )
    #mean_absolute_error (about 94.81%,93.26% accuracy, 69.69% with sliding window and new data)
    
    #https://keras.io/optimizers/
    #Check this website for optimizers for the model
    
    #scores = model.evaluate(test_X, test_Y, verbose=1)
    #print("Accuracy: %.2f%%" % (scores[1]*100))

    return model, X, Y, test_X, test_Y, train_X, train_Y

model, X, Y, test_X, test_Y, train_X, train_Y = lstm_sliding(df)

scores = model.evaluate(test_X, test_Y, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

yhat = model.predict(X, verbose=1)
yhat = np.around(yhat)
score = accuracy_score( yhat,Y)
print("Accuracy is", score*100 ,"%")

#Make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X)) #[samples, timesteps, features]
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

#forward = add new row of data each time you want to geed it into the model
#SEPERATE THE MODEL FOR INDIVIDUAL ATHLETE AS WELL, OR TRAIN SEPERATE MODELS FOR DIFFERENT ATHLETES
#USE OLD DATA FILE TO TRAIN, NEW DATA FILE TO TEST AND VALIDATE
#%%#-----------------------Seperate athletes in different dataframes-----------------------------------------------------------------------------
#For the purpose of easing the process of slinding window, seperate dataframe
#based on each individual athletes

#Train models seperatly

valuecounts1 = df["Participantsname"].value_counts()
athlete_name = valuecounts1.iloc[:].index.tolist()
dict_of_athletes = dict(tuple(df.groupby("Participantsname")))
del valuecounts1

for i in range(len(athlete_name)):
    name = athlete_name[i]
    data_personalized = dict_of_athletes[name]
    data_personalized = series_to_supervised(data_personalized)
    data_personalized = verrekt_fix(data_personalized):
    
    Y_values = data_personalized[data_personalized.columns[len(data_personalized.columns)-1]].values
    X_values = data_personalized.drop(data_personalized.columns[-1],axis=1)
    X_values = X_values.values

    train_X = X_values.reshape((X_values.shape[0], 1, X_values.shape[1]))
    train_Y = Y_values
        
    #Build model
    #Model Parameters
    model = Sequential()
    model.add(LSTM(5))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=20, batch_size=2, verbose=1)
    print(model.summary())
    
    scores = model.evaluate(train_X, train_Y, verbose=1)
    print("Accuracy of " name "is: %.2f%%" % (scores[1]*100))
    
#%%#-----------------------Tuning----------------------------
#% seems to be working for ANN - data forward

# Use scikit-learn to grid search the dropout rate
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', learn_rate=0.01, momentum=0, init_mode='uniform', activation='relu', dropout_rate=0.0, weight_constraint=0, neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=70, kernel_initializer=init_mode, activation=activation, kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
model = KerasClassifier(build_fn=create_model, verbose=2)
# define the grid search parameters
batch_size = [40, 80, 100]
epochs = [50, 100]
optimizer = ['SGD', 'Adam', 'Adamax']
learn_rate = [0.01, 0.1]
momentum = [0.0, 0.2, 0.4]
init_mode = ['uniform', 'normal', 'zero']
activation = ['softmax', 'tanh', 'sigmoid']
weight_constraint = [5, 7]
dropout_rate = [0.5, 0.7]
neurons = [25, 30, 40]
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, learn_rate=learn_rate, momentum=momentum, init_mode=init_mode, activation=activation, dropout_rate=dropout_rate, weight_constraint=weight_constraint, neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)


X,y = walking_forward(df,2041,0,0.0)
colum_names = ["best_params","best_score" ]
best = pd.DataFrame(columns=colum_names)
best_params = []
best_score = []

import time
start_time = time.time()

for i in range(1,len(X)):
        train_X = X[i]
        train_y = y[i]
        
        #train_X = train_X.values       
        #train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        #train_y = train_y.values
        
        grid_result = grid.fit(train_X, train_y)
        # summarize results
        best_params.append(grid_result.best_params_)
        best_score.append(grid_result.best_score_)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))

elapsed_time = time.time() - start_time
print("This simulation ran for",elapsed_time, "seconds, around",elapsed_time/60,"minutes.")


best["best_params"] = best_params
best["best_score"] = best_score

best = best.sort_values(by=["best_score"],ascending=False)

best.to_csv(r'C:\Users\Lian\OneDrive\Phase 4(With new data) - 27.03.2020\Script\Parameters\'ANN_forward_chaining.csv')

#%%#-----------------------ANN forward chaining-specialised model----------------------------------------------
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD, Adam, Adamax
import matplotlib.pyplot as plt

X,y,test_X,test_Y = walking_forward(df,2041,0,0.0)

opt = SGD(lr=0.1, momentum = 0.0)
opt = Adam(lr = 0.01)
opt = Adamax(lr=0.01)

import time
start_time = time.time()

model = Sequential()
model.add(Dense(25, input_dim=70, kernel_initializer='uniform', activation='softmax', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer='uniform', activation='softmax'))
# Compile model
model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['accuracy'])

for i in range(1,len(X)):
    train_X = X[i]
    train_Y = y[i]
    
    if i == 2:
        model.fit(train_X, train_Y, epochs=50, batch_size=40, validation_data=(test_X,test_Y), verbose=1)
    else: 
        history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y))

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = model.predict(test_X)
predictions = np.around(predictions)
score = accuracy_score(test_Y ,predictions)
print("Accuracy is", score*100 ,"%")

elapsed_time = time.time() - start_time

colum_name = ["test","predicted" ]
compare = pd.DataFrame(columns=colum_name)

compare["test"] = test_Y
compare["predicted"] = predictions

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

x_axis = np.arange(len(predictions))

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
plt.figure(figsize=(9,3))
plt.plot(x_axis, test_Y,'o',markersize=2,markeredgecolor='gray')
plt.plot(x_axis, predictions,'x',markersize=1)
plt.title('Model Prediction VS Actual Values')
plt.ylabel('Class')
plt.xlabel('Time Step')
plt.legend(['Actual','Predicted'], loc='upper left')
plt.rcParams["figure.figsize"] = [16,9]
plt.show()







#%% seems to be working, implimenting lstm for data sliding

# Use scikit-learn to grid search the dropout rate
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', learn_rate=0.01, momentum=0, init_mode='uniform', activation='relu', dropout_rate=0.0, weight_constraint=0, neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=354, kernel_initializer=init_mode, activation=activation, kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
model = KerasClassifier(build_fn=create_model, verbose=2)
# define the grid search parameters
batch_size = [40, 80, 100]
epochs = [50, 100]
optimizer = ['SGD', 'Adam', 'Adamax']
learn_rate = [0.01, 0.1]
momentum = [0.0, 0.2, 0.4]
init_mode = ['uniform', 'normal', 'zero']
activation = ['softmax', 'tanh', 'sigmoid']
weight_constraint = [5, 7]
dropout_rate = [0.5, 0.7]
neurons = [25, 30, 40]

param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, learn_rate=learn_rate, momentum=momentum, init_mode=init_mode, activation=activation, dropout_rate=dropout_rate, weight_constraint=weight_constraint, neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3) 

colum_names = ["best_params","best_score" ]
best = pd.DataFrame(columns=colum_names)
best_params = []
best_score = []
    
Y = data_sliding[data_sliding.columns[len(data_sliding.columns)-1]].values
X_values = data_sliding.drop(data_sliding.columns[-1],axis=1)

X_values = X_values.values
X = X_values.astype(str).astype(float)

split_value = 1500;
test_X = X[split_value:,:]
test_Y = Y[split_value:]

train_X = X[:split_value,:]
train_Y = Y[:split_value]

import time
start_time = time.time()

grid_result = grid.fit(train_X, train_Y)

elapsed_time = time.time() - start_time
print("This simulation ran for",elapsed_time, "seconds, around",elapsed_time/60,"minutes.")

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
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

best.to_csv(r'C:\Users\Lian\OneDrive\Phase 4(With new data) - 27.03.2020\Script\Parameters\'ANN_data_sliding.csv')

#%%#-----------------------ANN data sliding-specialised model----------------------------------------------

import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.metrics import accuracy_score

Y = data_sliding[data_sliding.columns[len(data_sliding.columns)-1]].values
X_values = data_sliding.drop(data_sliding.columns[-1],axis=1)

X_values = X_values.values
X = X_values.astype(str).astype(float)

split_value = 1500;
test_X = X[split_value:,:]
test_Y = Y[split_value:]

train_X = X[:split_value,:]
train_Y = Y[:split_value]

import time
start_time = time.time()

model = Sequential()
model.add(Dense(40, input_dim=354, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(7)))
model.add(Dropout(0.7))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='mean_absolute_error', optimizer='Adamax', metrics=['accuracy'])

history = model.fit(train_X, train_Y, epochs=100, batch_size=40,validation_data=(test_X,test_Y), verbose=1)


print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = model.predict(test_X)
predictions = np.around(predictions)
score = accuracy_score(test_Y ,predictions)
print("Accuracy is", score*100 ,"%")

elapsed_time = time.time() - start_time

colum_name = ["test","predicted" ]
compare = pd.DataFrame(columns=colum_name)

compare["test"] = test_Y
compare["predicted"] = predictions


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

x_axis = np.arange(len(predictions))

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
plt.figure(figsize=(9,3))
plt.plot(x_axis, test_Y,'o',markersize=2,markeredgecolor='gray')
plt.plot(x_axis, predictions,'x',markersize=1)
plt.title('Model Prediction VS Actual Values')
plt.ylabel('Class')
plt.xlabel('Time Step')
plt.legend(['Actual','Predicted'], loc='upper left')
plt.rcParams["figure.figsize"] = [16,9]
plt.show()


