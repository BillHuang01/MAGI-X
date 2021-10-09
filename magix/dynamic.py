import torch
import numpy as np

'''
    Dynamic Module:
    This module is for setting up the ODE dynamic. 
'''

torch.set_default_dtype(torch.double)

class odeModule(torch.nn.Module):
    def __init__(self, fOde, theta):
        super().__init__()
        self.f = fOde
        if (not torch.is_tensor(theta)):
            theta = torch.tensor(theta).double()
        self.theta = torch.nn.Parameter(theta)

    def forward(self, x):
        return (self.f(self.theta, x))

class nnModule(torch.nn.Module):
    # Neural Network with Constraint on Output
    def __init__(self, nNodes, bc=None):
        super().__init__()
        # define layers
        layers = []
        for i in range(len(nNodes)-1):
            layers.append(torch.nn.Linear(nNodes[i],nNodes[i+1],bias=True))
        self.layers = torch.nn.ModuleList(layers).double()
        # define output constraint
        if (bc is None):
            bc = np.repeat([[-np.inf,np.inf]], nNodes[-1], axis = 0)
        self.output_layer = []
        for i in range(bc.shape[0]):
            if (bc[i,0] == -np.inf and bc[i,1] == np.inf):
                self.output_layer.append(lambda x: x) # identity function
            elif (bc[i,0] == -np.inf and bc[i,1] == 0):
                self.output_layer.append(lambda x: -torch.exp(x))
            elif (bc[i,0] == 0 and bc[i,1] == np.inf):
                self.output_layer.append(lambda x: torch.exp(x))
            else:
                self.output_layer.append(lambda x,lb=bc[i,0],ub=bc[i,1]: lb+(ub-lb)*torch.sigmoid(x))

    def forward(self, x):
        if (len(x.size())==1):
            # reshape vector to row length 1 matrix
            x = torch.reshape(x,(1,-1)) 
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[i](x))
            # x = torch.tanh(self.layers[i](x))
        x = self.layers[-1](x)
        for i in range(len(self.output_layer)):
            x[:,i] = self.output_layer[i](x[:,i])
        return (x)

    def reset(self):
        for layer in self.layers:
            layer.reset_parameters()
