import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess  #this package exists on my computer only. Do not copy line 1,3,4,10 of this code. 

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels

features_train, features_test, labels_train, labels_test = preprocess() #this is what preprocess does. returns training and testing variables

#important part starts from here.
from sklearn import svm 

clf = svm.SVC(C=10000,kernel = "rbf")
#t = time()

clf.fit(features_train,labels_train)
#print("Training time : ",round(time()-t,3)," s") 

#t2 = time()
pred = clf.predict(features_test[:])
#print("Training time : ",round(time()-t,3)," s")

"""
  one = 0
  for i in range(len(pred)):
      if pred[i]==1:
          one+=1

  print(one)
"""
