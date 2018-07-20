#!/usr/bin/python

""" 
This is the code to accompany the Lesson 2 (SVM) mini-project.

Use a SVM to identify emails from the Enron corpus by their authors:    
Sara has label 0
Chris has label 1
"""
    
import sys
from time import
sys.path.append("../tools/")
from email_preprocess import preprocess 

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn import svm 

clf = svm.SVC(C=10000,kernel = "rbf")
t = time()


clf.fit(features_train,labels_train)
#print("Training time : ",round(time()-t,3)," s")

#t = time()
pred = clf.predict(features_test[:])
#print("Training time : ",round(time()-t,3)," s")

one = 0
for i in range(len(pred)):
    if pred[i]==1:
        one+=1

print(one)
