'''
Training a random dataset using categorical cross entropy which requires one-hot-labeling.
Quoted from Keras documentation:-
"Note: when using the categorical_crossentropy loss, your targets should be in categorical format (e.g. if you
have 10 classes, the target for each sample should be a 10-dimensional vector that is all-zeros except for a 1 at
the index corresponding to the class of the sample). In order to convert integer targets into categorical targets, 
you can use the Keras utility 'to_categorical:' "

Used a softmax activation to calculate the score of each class. 
''' 

import keras
from keras.models import Sequential
import numpy as np

from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(32,input_shape=(100,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer = "rmsprop",
             loss = "categorical_crossentropy",
             metrics = ['accuracy'])

data_train = np.random.random((100,100))
label = np.random.randint(10,size=(100,1))
data_test = np.random.random((10,100))
pred = np.random.randint(10,size=(10,1))

one_hot_labels = keras.utils.to_categorical(label, num_classes = 10)
one_hot_pred = keras.utils.to_categorical(pred,num_classes = 10)

model.fit(data_train,one_hot_labels,epochs = 10,batch_size = 10)

score = model.evaluate(data_test,one_hot_pred, batch_size = 10)

print(score,model.metrics_names)
