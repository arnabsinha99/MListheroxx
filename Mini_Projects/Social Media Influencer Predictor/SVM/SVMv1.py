 
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:39:35 2019

@author: ROHIT
"""



#No features are removed




#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

#reading the train files
data=pd.read_csv("train.csv")
 
y=data[['Choice']]
data.describe()

 
list(data.columns)

plt.plot(data[['A_follower_count']],data[['B_following_count']],'+',color='r')



X=data.drop(columns='Choice') 
X_test=pd.read_csv("test.csv")
b=X_test 




#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)
b = sc.transform(b)
 
#importing the SVM model 
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0,probability=True)
classifier.fit(X,y)

#reading the submission files
bp=pd.read_csv("gopal.csv")
gh=bp.iloc[:,1].values
gh=np.insert(gh,0,0.315025203)

for i in range(0,5952):
    if gh[i]<=0.5:
        
        gh[i]=0
        
for i in range(0,5952):
    if gh[i]>0.5:
        
        gh[i]=1

#predicting the values on test set
y_pred = classifier.predict(b)
auc = roc_auc_score(gh, y_pred)


y_pred1 = classifier.predict_proba(b)
rts=y_pred1[:,1]


#importing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(gh, y_pred)



