import numpy as np

class NueralNetwork(object):
    def __init__(self, weights, bias, features, outputs, alpha, iterations):
        self.weights    = weights
        self.features   = features
        self.outputs    = outputs
        self.alpha      = alpha
        self.iterations = iterations
        self.bias       = bias

    def __hardlim(self, i):
        """converts input into a one or a zero,
        must be vectorized to run over numpy array"""
        if i >= 0.5:
            return 1
        else:
            return 0

    def train(self):
        output = None
        for i in range(0, self.iterations):
            # calculate the output
            output = np.dot(self.features, self.weights.T) + self.bias
            # pass to hardlim function
            vhardlim = np.vectorize(self.__hardlim)
            output = vhardlim(output)
            # calculate error and check if outputs match expected
            # output values
            error = np.absolute( outputs - output )
            # check if error is zero, if not update weights and bias
            if not np.all( error==0 ):
                self.weights = self.weights + self.alpha * error * self.features.T
                self.bias = self.bias + self.alpha * error

        return output

weights = np.array([[1,0,3]])
bias = 4
alpha = 0.2
features = np.array([[1,2,3], [2,1,6], [4,2,1]])
outputs = np.array([1,0,0])
iterations = 9

nn = NueralNetwork(weights, bias, features, outputs, alpha, iterations)
print( nn.train() )

