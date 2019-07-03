import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.misc as sm
import numpy as np




data=pd.read_csv('train.csv')
X_test=pd.read_csv('test.csv')

X=data.drop(columns=['Choice'])
a=list(X.columns)
data=data.drop(columns=['Choice'])

 
#feature scaling on training datasets
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
data=sc.fit_transform(data)

data=pd.DataFrame(data)
 
columns3=data.iloc[:,[3,4,8,14,15,19]].values
columns3=pd.DataFrame(columns3)

columns3[0]

#column which are independent with others
columns1=['A_mentions_received','A_retweets_received','A_network_feature_1','B_mentions_received','B_retweets_received','B_network_feature_1']

     
i=1

f=plt.figure(figsize=(15,15))
bt=np.arange(0,23)

for i in range(0,6):
    f=plt.figure(figsize=(15,15))
    for y in range(0,21):
      
        f=plt.subplot(6,4,y+1)
        plt.plot(columns3[i],data[y],'+',color='r')
        plt.xlabel(columns1[i])
        plt.ylabel(a[y])
        
    plt.savefig(f'images{i}.png')
      
  
          
            
              
              
                  
              
              
              
             
    
    