# -*- coding: utf-8 -*-
"""Animal_pred_XGBoost_ver1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17996Sl4YgA57Qm3Nk56eYje1y_JX8TGA
"""

''' The following model using XGBoost gave me 39.5 % accuracy on first try. '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import io
from google.colab import files

def transform(df,str,bit):
    
  map_outcome = {}
  for x,y in enumerate(df[str].unique()):
    map_outcome[y] = x
  outcome_type = [map_outcome[x] for x in df[str]]
  if bit==0:
    return outcome_type
  elif bit==1:
    return map_outcome

def inv_transform(df,str,predicted):
  
  map_outcome = transform(df,str,1)
  keys = map_outcome.keys()
  vals = map_outcome.values()
  mapinv = {}
  #print(type(keys),type(vals),type(mapinv))
  for i,j in zip(vals,keys):
    mapinv[i] = j
  
  pred = predicted.tolist()
  ret = []
  
  for i in pred:
    ret.append(mapinv[i])
    
  return ret

def preprocessing(df):
  
  df2 = []

  #taking the first color
  
  changed = {} # map to store the unique keys and resp. values of each column of type string

  for str in df['color']:
    str2 = str.split('/')[0]
    df2.append(str2)

  df.drop('color',axis = 1)
  df['color'] = df2

  df.drop('age_upon_intake_(days)',axis = 1,inplace = True)

  #change the column from range to difference between max and min
  df2 = []
  df3 = []
  for str in df['age_upon_intake_age_group']:
    str = str[1:len(str)-1].split(',')
    df2.append(str[0])
    df3.append(str[1])

  df.drop('age_upon_intake_age_group',axis=1,inplace = True)
  df['age_upon_intake_left_limit'] = [float(i) for i in df2]
  df['age_upon_intake_right_limit'] = [float(i) for i in df3]
  
  df2 = []
  df3 = []
  for str in df['age_upon_outcome_age_group']:
    str = str[1:len(str)-1].split(',')
    df2.append(str[0])
    df3.append(str[1])

  df.drop('age_upon_outcome_age_group',axis=1,inplace = True)
  df['age_upon_outcome_left_limit'] = [float(i) for i in df2]
  df['age_upon_outcome_right_limit'] = [float(i) for i in df3]
  
  a = df['age_upon_outcome_right_limit'][0]
  print(type(a))
  #print(type(float(df['age_upon_outcome_right_limit'][0]))) it prints float
  
  df.drop(['intake_datetime','date_of_birth','time_in_shelter','intake_monthyear','outcome_datetime','outcome_monthyear'],axis=1,inplace = True) #it is unnecessary as info is already gathered in adjacent columns
  
  # to check difference between intake and outcome sex
  count=0

  for a,b in zip(df['sex_upon_intake'],df['sex_upon_outcome']):
    if(a!=b):
      count = count + 1

  from collections import Counter
  #Counter(df['sex_upon_intake'])

  df['sex_upon_intake'].fillna(value = 'Intact Male',inplace = True)

  # tbh count = 0

  '''
  for i in df['sex_upon_outcome']:
    if pd.isnull(i):
      print(count)
    count = count + 1
  print(count)
  '''
  #since Neutured Male was the majority one
  # tbh df.loc[6523,'sex_upon_outcome'] = "Neutered Male" #do not use df.loc[6523]['sex_upon_outcome']

  # tbh df.isnull().sum().sum() # ok so now no null values exist

  column_names = df.columns.values

  j=0
  for i in df.loc[0]:
    if isinstance(i,__builtins__.str):
      changed[column_names[j]] = transform(df,column_names[j],0)
    j=j+1

  #print(changed)

  for i in changed:
    df.drop(i,axis = 1,inplace = True)
    df[i] = changed[i]

  
  #So, now I have changed all string columns to integer columns. Now onto model deployment

  #print([ type(x) for x in df.loc[0]])  #got all as float
  return df,changed

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

#so I imported the datasets from my google drive folder
from google.colab import drive
drive.mount('/content/gdrive',force_remount = True)
df = pd.read_csv('/content/gdrive/My Drive/Datasets/train.csv',sep = None)

changed = {}

y_train = df['outcome_type']
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)

trainingdf,changed = preprocessing(df) #df after preprocessing

x_train = trainingdf.drop('outcome_type',axis = 1)

model = XGBClassifier()
model.fit(x_train.values,encoded_Y)

dff = pd.read_csv('/content/gdrive/My Drive/Datasets/test.csv',sep = None)

changed_test = {}
animal_id_outcome = dff['animal_id_outcome']
testingdf,changed_test = preprocessing(dff)


pred = model.predict(testingdf.values)
# print(predicted)

pred = encoder.inverse_transform(pred)
dict = {'animal_id_outcome':animal_id_outcome,'outcome_type':pred.tolist()}
final = pd.DataFrame(data = dict)
print(final) # to check output