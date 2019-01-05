#In [1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
type(cancer)
print(cancer['DESCR'])
cancer['feature_names']
df = pd.DataFrame(cancer['data'],columns = cancer['feature_names'])
df.info()
cancer['target']
df_target = pd.DataFrame(cancer['target'],columns = ['Cancer'])
df.head()
from sklearn.model_selection import train_test_split
df_target.shape
(np.ravel(df_target)).shape
X_train, X_test, y_train, y_test = train_test_split(df, np.ravel(df_target), test_size=0.30, random_state=102)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
from sklearn.metrics import accuracy_score
grid_predictions
y_test
accuracy_score(y_test,grid_predictions)