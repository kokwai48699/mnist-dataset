
# coding: utf-8

# In[63]:

#import standard libraries
import random 
import numpy as np

#import the datasets using keras for mnist
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[64]:

#print the dimensions of the datasets 
print x_train.shape
print y_train.shape


# In[143]:

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z)) #sigmoid function

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z)) # Derivatives of the sigmoid function

    #create layers and set the biases and weights to random values 
class network(object):
    def __init__(self,sizes):
        self.layers_num = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
        #feed forward process 
    def feedforward(self, activation):
        for b, w in zip(self.biases, self.weights):
            activation = sigmoid(np.dot(w, activation)+b)
        return activation
    
    def mini_batch(self, m_batch, l_rate):
        zero_b = [np.zeros(b.shape) for b in self.biases]
        zero_w = [np.zeros(w.shape) for w in self.biases]
    
        for x, y in m_batch:
            delta_zero_b, delta_zero_w = self.backprop(x,y)
            zero_b = [nb+dnb for nb, dnb in zip(zero_b, delta_zero_b)]
            zero_w = [nw+dnw for nw, dnw in zip(zero_w, delta_zero_w)]
        
        self.weights = [w-(l_rate/len(m_batch))*nw for w, nw in zip(self.weights, zero_w)]
        self.biases = [b-(l_rate/len(m_batch))*nb for b, nb in zip(self.biases, zero_b)]
    
    def cost_function(self, out_activation, y):
        return (out_activation - y)
    
    #Set the initial biases and weights as zeros for each tuple 
    def backprop(self, x, y):
        zero_b = [np.zeros(b.shape) for b in self.biases]
        zero_w = [np.zeros(w.shape) for w in self.biases]

        activation = x 
        activations = [x]
        zs = []
    
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
            #calculate the error in the output layer and then backpropagation 
        error = self.cost_function(activation[-1], y) * sigmoid_prime(zs[-1])
        zero_b[-1] = error
        zero_w[-1] = np.dot(error, activations[-2].transpose())
    
        for l in xrange(2, self.layers_num):
                z = zs[-l]
                sp = sigmoid_prime(z)
                error = np.dot(self.weights[-l+1].transpose(), error) * sp
                zero_b[-l] = error
                zero_w[-l] = np.dot(error, activations[-l-1].transpose())

        return (zero_b, zero_w)
    
    #return the correct output results from the test inputs
    def train(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    #Training the neural network by using the mini-batch of size 10
    def train_network(self, training_data, epochs, mini_batch_size, l_rate, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for m_batch in mini_batches:
                self.mini_batch(m_batch, l_rate)
            if test_data:
                print "Epoch {0}: {1} / {2}, learning_rate:%.3f".format(
                    j, self.evaluate(test_data), n_test, l_rate)
            else:
                print "Epoch {0} complete".format(j)




# In[144]:

nn = network([784, 100, 50, 10])


# In[145]:

for w in nn.weights:
    print w.shape
train_arr = []
for x,y in zip(x_train,y_train):
    y_enc = np.array([i==y for i in range(10)]).reshape((10,1))
    x = x.reshape((784,1))
    train_arr.append([x,y_enc])


# In[137]:

nn.train_network(train_arr, 10, 100, 0.003)


# In[146]:

c = 0
for x,y in zip(x_test,y_test):
    y_pred = nn.feedforward(x.reshape((784,1)))
    print(y_pred)
    y_pred = np.argmax(y_pred)
    print(y,y_pred)
    c += 1
    if c == 10:
        break


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



