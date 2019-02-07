# Simple code to implement NN in Keras. 


from keras.models import Sequential
import numpy as np

from keras.layers import Dense, Activation

'''model = Sequential([
 Dense(32, input_shape = (784,)),
 Activation('relu'),
 Dense(16),
 Activation('sigmoid') 
])'''
#The above is same as the one below

model = Sequential()
model.add(Dense(32, input_shape=(100,)))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer = "sgd",
             loss = "mse",
             metrics = ['accuracy'])

data_train = np.random.random((100,100))
label= np.random.randint(2,size=(100,1))
data_test = np.random.random((10,100))
data_pred = np.random.randint(2,size = (10,1))

model.fit(data_train,label,epochs = 10, batch_size = 10)

score = model.evaluate(data_test,data_pred, batch_size = 10)

print(score)
