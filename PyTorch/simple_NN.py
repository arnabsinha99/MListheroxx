import torch as tor
import torch.nn as nn

X = tor.tensor(([1,2],[4,5],[2,9]), dtype = tor.float)
y = tor.tensor(([92],[89],[60]), dtype = tor.float)

xPredicted = tor.tensor(([4,8]),dtype = tor.float)
X_max, _ = tor.max(X,0)

xPredicted_max, _ = tor.max(xPredicted, 0)

X = tor.div(X, X_max)
xPredicted = tor.div(xPredicted, xPredicted_max)
y = y / 100 # max test score is 100

''' now comes the asli code '''

class neuralnetwork(nn.Module):
  def __init__(self, ):
    super(neuralnetwork, self).__init__()
    
    #pameters to be declared
    
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenLayers = 3
    
    self.W1 = tor.randn(self.inputSize,self.hiddenLayers) #2 * 3 matrix for weights for i/p layers
    self.W2 = tor.randn(self.hiddenLayers, self.outputSize) #1* 3 matrix for weights for o/p layers
    
  def forward(self, X):
    
    self.z = tor.matmul(X, self.W1)
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = tor.matmul(self.z2, self.W2)
    
    out = tor.sigmoid(self.z3) #activation
    
    return out
 
  def sigmoid(self, x):
    
    a = 1/(1+tor.exp(-x))
    return a
  
  def sigmoid_d1(self,x):
    
    return x * (1-x)
  
  def backward(self,X,y,o):
    
    self.out_error = y-o
    self.delta = self.sigmoid_d1(o) * self.out_error  #  h'(x) * (h(x) - y)
    self.hidd_error = tor.matmul(self.delta,tor.t(self.W2))
    self.delta2 = self.sigmoid_d1(self.hidd_error) * self.hidd_error
    
    self.W1 += tor.matmul(tor.t(X), self.delta2)
    self.W2 += tor.matmul(tor.t(self.z2), self.delta)
    
  def train(self,X,y):
    
    o = self.forward(X)
    self.backward(X,y,o)  
    
  def save_weights(self,model):
    
    tor.save(model,"NN")
    
  def predict(self):
    
    print("The predicted data based on trained weights are: ")
    print ("Input (scaled): \n" + str(xPredicted))
    print ("Output: \n" + str(self.forward(xPredicted).detach().item()*100))    


NN = neuralnetwork()
for i in range(1000):  # trains the NN 1,000 times
    #print ("#" + str(i) + " Loss: " + str(tor.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
    
NN.save_weights(NN)
NN.predict()    
