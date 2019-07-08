
#NO FEATURES ARE REMOVED

#importing the libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


#reading the data to train our model
data=pd.read_csv("train.csv")
 
y=data[['Choice']]
data.describe()

 
 


X=data.drop(columns=['Choice']) 
X_test=pd.read_csv("test.csv")
b=X_test 



#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)
b = sc.transform(b)
 
#importing random forest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =100,criterion = 'entropy', random_state = 0)
classifier.fit(X, y)

#reading the submission firl to measure accuracy
bp=pd.read_csv("gopal.csv")
gh=bp.iloc[:,1].values
gh=np.insert(gh,0,0.315025203)

#converting the sumbision file to binary
for i in range(0,5952):
    if gh[i]<=0.5:
        
        gh[i]=0
        
for i in range(0,5952):
    if gh[i]>0.5:
        
        gh[i]=1
        
gh=pd.DataFrame(gh)      
gh[0] = gh[0].apply(np.int64)

#measuring the predicted value on test set
y_pred = classifier.predict(b)
auc = roc_auc_score(gh, y_pred)


y_pred1 = classifier.predict_proba(b)
rts=y_pred1[:,1]

#imporitng tha confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(gh, y_pred)









