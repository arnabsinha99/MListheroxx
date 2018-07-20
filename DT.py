#!/usr/bin/python

""" 
    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

##  print len(features_train[0])   ##to print the number of features of a particular sample

##  print features_train.shape    ##to print the dimensions of the dataframe

clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train,labels_train)

pred = clf.predict(features_test)

acc = accuracy_score(pred,labels_test)

#print "Accuracy is ",acc
