# Fábrica
######################################################################
class ActivationFactory():
    def __init__(self):
        super().__init__()

    def getActivation(self,parameters: dict):
        if parameters['ACTIVATION'] == 'RELU':
            return Relu(parameters['BIAS'])
######################################################################

# Clases
######################################################################
class Activation():
    def __init__(self):
        super().__init__()
        
    def activation(self, x):
        return 'instanceless'

    def applyActivation(self,image):
        import numpy as np
        imageshape = image.shape
        result = np.array(list(map(self.activation,image.flatten())))
        return result.reshape(imageshape)

class Relu(Activation):
    def __init__(self, bias=0):
        super().__init__()
        self.bias = bias

    def activation(self, x):
        return max(0,x + self.bias)
######################################################################