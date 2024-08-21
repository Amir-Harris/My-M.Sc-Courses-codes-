import numpy as np
import random
import time
import json

#activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return x*(1-x)

class NN:
    def __init__(self, inputs):
        random.seed(1)
        #to get the same random number
        self.inputs = inputs
        self.l = len(self.inputs) #this is 4
        self.li = len(self.inputs[0]) # this is 2 ([0,0])
        #np.array([[ 0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]])
        self.wi = np.random.random((self.li, self.l))
        #initial weights
        #print self.wi
        #np.array([[0.0], [0.0], [0.0], [0.0]])
        self.wh = np.random.random((self.l,1))
        #hidden layer weights
        #print self.wh

    def think(self, inp):
        s1 = sigmoid(np.dot(inp, self.wi))
        s2 = sigmoid(np.dot(s1, self.wh))
        return s2



    def train(self, inputs, outputs):

        l0 = inputs
        l1 = sigmoid(np.dot(l0, self.wi))
        l2 =  sigmoid(np.dot(l1, self.wh))
        #calculating errors
        l2_err = outputs - l2
        l2_delta = np.multiply(l2_err, sigmoid_der(l2))

        l1_err = np.dot(l2_delta, self.wh.T)
        l1_delta = np.multiply(l1_err, sigmoid_der(l1))

        #updating the weights
        self.wh += np.dot(l1.T, l2_delta)
        self.wi += np.dot(l0.T, l1_delta)


inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
outputs = np.array([[0], [1], [1], [0]])
out = []
data = {"Initial-input" : None, "output" : None, "Time" : None, "Epochs": None}
stime= time.time()
n = NN(inputs)
#print "Before training" ,n.think(inputs)
ip = n.think(inputs)
data["Initial-input"] = str(ip)
n.train(inputs,outputs)
count = 1
op = n.think(inputs)
#accepted error is 0.01
while (outputs[1]-op[1] > [0.005] and outputs[2]-op[2] > [0.005]):
    n.train(inputs,outputs)
    op = n.think(inputs)
    count +=1
etime = time.time() - stime
#print "After training", op, etime, "\n"
data["output"] = str(op)
data["Epochs"]= count
data["Time"]= etime
outfile = 'mlp.json'
out.append(data)
json.dump(out,open(outfile, 'a'))
print ("FINAL weight of hidden node: " , n.wi)
print ("FINAL weight of output node: " , n.wh)
