# Import Statements
import numpy as np

# Class Definitions
class NeuralNetwork(object) :

    def __init__(self) :
        # Define Hyperparameters
        self.hidden_layer_size = 3
        self.input_layer_size = 2
        self.output_layer_size = 1

        # Create Random Weight Matrices to Start With
        self.W1 = np.random.randn(self.input_layer_size,
            self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size,
            self.output_layer_size)

    def forward(self, x) :
        #Propogate inputs through network
        self.z2 = np.dot(x, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        y_hat = self.sigmoid(self.z3)
        return y_hat

    def sigmoid(self,z) :
        return  1 / ( 1 + np.exp(-z))

x = np.array(([3,5], [5,1], [10,2]), dtype = float)
y = np.array(([75], [82], [93]), dtype = float)

x_norm = x/np.amax(x, axis = 0)
y_norm = y / 100 # max test score is 100

nn = NeuralNetwork()
y_estimates = nn.forward(x_norm)

print y_estimates
print y_norm

#####TESTING####TESTING####TESTING####TESTING####TESTING####TESTING####TESTING##
print '-' * 10
print x
print y
print x_norm
print y_norm
