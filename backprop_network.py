"""

backprop_network.py

"""



import random

import numpy as np

from scipy.special import softmax

import math



class Network(object):



    def __init__(self, sizes):

        """The list ``sizes`` contains the number of neurons in the

        respective layers of the network.  For example, if the list

        was [2, 3, 1] then it would be a three-layer network, with the

        first layer containing 2 neurons, the second layer 3 neurons,

        and the third layer 1 neuron.  The biases and weights for the

        network are initialized randomly, using a Gaussian

        distribution with mean 0, and variance 1.  Note that the first

        layer is assumed to be an input layer, and by convention we

        won't set any biases for those neurons, since biases are only

        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x)

                        for x, y in zip(sizes[:-1], sizes[1:])]



    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,

            test_data):

        """Train the neural network using mini-batch stochastic

        gradient descent.  The ``training_data`` is a list of tuples

        ``(x, y)`` representing the training inputs and the desired

        outputs.  """

        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))

        n = len(training_data)
     
        # for clause b
        train_accuracy = np.zeros(epochs)
        test_accuracy = np.zeros(epochs) 
        train_loss = np.zeros(epochs)
        # until here

        for j in range(epochs):

            random.shuffle(training_data)

            mini_batches = [

                training_data[k:k+mini_batch_size]

                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:

                self.update_mini_batch(mini_batch, learning_rate)

            print ("Epoch {0} test accuracy: {1}".format(j, self.one_label_accuracy(test_data)))
            
            # for clause b
            #train_accuracy[j] = self.one_hot_accuracy(training_data)
            test_accuracy[j] = self.one_label_accuracy(test_data)
            #train_loss[j] = self.loss(training_data)

        #print ("Epoch {}: test accuracy: {}".format(j, self.one_label_accuracy(test_data)))
               
        print ("Finished SGD - test accuracy: {}".format(self.one_label_accuracy(test_data)))
        return train_accuracy, train_loss, test_accuracy


    def update_mini_batch(self, mini_batch, learning_rate):

        """Update the network's weights and biases by applying

        stochastic gradient descent using backpropagation to a single mini batch.

        The ``mini_batch`` is a list of tuples ``(x, y)``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw

                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (learning_rate / len(mini_batch)) * nb

                       for b, nb in zip(self.biases, nabla_b)]



    def backprop(self, x, y):
        """The function receives as input a 784 dimensional 

        vector x and a one-hot vector y.

        The function should return a tuple of two lists (db, dw) 

        as described in the assignment pdf. """
                
        v, z, activation_derivatives = self.forward_pass(x)

        deltas = [None] * len(self.weights) 
        loss_derivatives_b = [None] * len(self.weights)  
        loss_derivatives_W = [None] * len(self.weights)  

        delta_L = self.loss_derivative_wr_output_activations(v[-1], y) # delta_L = z_L - y
        # update last entry 
        deltas[-1] = delta_L
        loss_derivatives_b[-1] = delta_L
        loss_derivatives_W[-1] = delta_L @ np.transpose(z[-2]) # this product is a matrix 
        # update one before last entry
        deltas[-2] = np.transpose(self.weights[-1]) @ delta_L
        loss_derivatives_b[-2] = np.multiply(deltas[-2], activation_derivatives[-2]) 
        loss_derivatives_W[-2] = loss_derivatives_b[-2] @ np.transpose(z[-3]) # this product is a matrix 

        return self.backward_pass(deltas, activation_derivatives, z, loss_derivatives_b, loss_derivatives_W)


    def forward_pass(self, x): 
        """ calc v_l, z_l, z_l' for l = 1,...,L """
        
        v = []
        z = [np.copy(x)]
        activation_derivatives = []
        for i in range(len(self.weights)):
            v.append(self.weights[i] @ z[-1] + self.biases[i])
            z.append(relu(v[-1]))
            activation_derivatives.append(relu_derivative(z[-1]))
        return v, z, activation_derivatives

    def backward_pass(self, deltas, activation_derivatives, z, loss_derivatives_b, loss_derivatives_W):
        """ calc db, dw using deltas, activation_derivatives """
              
        for i in reversed(range(len(deltas) - 2)): # i = L-3,L-4,...,1,0
            deltas[i] = np.transpose(self.weights[i + 1]) @ loss_derivatives_b[i + 1] # loss_derivatives_b[i + 1] = z_i+1' o delta_i+1
            loss_derivatives_b[i] = np.multiply(deltas[i], activation_derivatives[i]) 
            loss_derivatives_W[i] = loss_derivatives_b[i] @ np.transpose(z[i]) 
        return (loss_derivatives_b, loss_derivatives_W)



    def one_label_accuracy(self, data):

        """Return accuracy of network on data with numeric labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), y)

         for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results)/float(len(data))



    def one_hot_accuracy(self,data):

        """Return accuracy of network on data with one-hot labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), np.argmax(y))

                          for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results) / float(len(data))



    def network_output_before_softmax(self, x):

        """Return the output of the network before softmax if ``x`` is input."""

        layer = 0

        for b, w in zip(self.biases, self.weights):

            if layer == len(self.weights) - 1:

                x = np.dot(w, x) + b

            else:

                x = relu(np.dot(w, x)+b)

            layer += 1

        return x



    def loss(self, data):

        """Return the CE loss of the network on the data"""

        loss_list = []

        for (x, y) in data:

            net_output_before_softmax = self.network_output_before_softmax(x)

            net_output_after_softmax = self.output_softmax(net_output_before_softmax)

            loss_list.append(np.dot(-np.log(net_output_after_softmax).transpose(),y).flatten()[0])

        return sum(loss_list) / float(len(data))



    def output_softmax(self, output_activations):

        """Return output after softmax given output before softmax"""

        return softmax(output_activations)



    def loss_derivative_wr_output_activations(self, output_activations, y):

        # Implement derivative of loss with respect to the output activations before softmax

        return self.output_softmax(output_activations) - y



def relu(z):

    """ Implement the relu function. """
    
    return (z > 0) * z 


def relu_derivative(z):

    """ Implement the derivative of the relu function."""
    
    return (z > 0) * 1.0 
