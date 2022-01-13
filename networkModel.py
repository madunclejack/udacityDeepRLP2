import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    '''
    Function that calculates the upper and lower limits for the
    hidden layer weight initialization. Limits are used to
    bound a uniform distribution so weights are not saturated
    or close to zero when the network starts to learn.
    This uses Xaviar initialization, which is useful for tanh
    activation functions
    '''
    fan_in = layer.weight.data.size()[0]
    fan_out = layer.weight.data.size()[1]
    lim = np.sqrt(6.0)/ np.sqrt(fan_in + fan_out)
    return (-lim, lim)

class Actor(nn.Module):
    """ Actor network model """
    
    def __init__(self, actionLims, state_size, action_size, seed, hiddenArray):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.stateSize = state_size
        self.actionSize = action_size
        self.outputActivation = F.tanh
        
        # Set up for compute device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
        
        
        # Calculate the action space min and max, and the
        # network min and max for the action scaling function
        self.env_min = torch.tensor(actionLims[0], device=self.device, dtype=torch.float32)
        self.env_max = torch.tensor(actionLims[1], device=self.device, dtype=torch.float32)
        self.nn_min = self.outputActivation(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = self.outputActivation(torch.Tensor([float('inf')])).to(self.device)
        
        self.hiddenLayers = hiddenArray
        self.NeuralNet = None
        self.output = None
        self.buildNetwork()
        self.resetParameters()
        
        
    def resetParameters(self):
        '''
        Resets the weights of the network.
        *functionName is a "methodcaller" operation in Python
        '''
        for layer in self.NeuralNet:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
    
    def rescaleAction(self, inputVal):
        # Calculates  y = mx + b type scaling
        print("rescale input")
        print(inputVal)
        return (inputVal - self.nn_min) * (self.env_max - self.env_min) / \
                (self.nn_max - self.nn_min ) + self.env_min
    
    def buildNetwork(self):
        self.NeuralNet = nn.ModuleList([nn.Linear(self.stateSize, self.hiddenLayers[0])])
        
        layerSizes = zip(self.hiddenLayers[:-1], self.hiddenLayers[1:])
        
        self.NeuralNet.extend([nn.Linear(h1, h2) for h1, h2 in layerSizes])
        self.output = nn.Linear(self.hiddenLayers[-1], self.actionSize)
        self.resetParameters()
    
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x
    
    def forward(self, state):
        x = self._format(state)
        for linearLayer in self.NeuralNet:
            x = F.relu(linearLayer(x))
        x = self.outputActivation(self.output(x))
        return x
    
class Critic(nn.Module):
    """ Critic Network Model """
    
    def __init__(self, state_size, action_size, seed, hiddenArray):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.stateSize = state_size
        self.actionSize = action_size
        self.hiddenLayers = hiddenArray
        self.NeuralNet = None
        self.output = None
        self.buildNetwork()
        self.resetParameters()
        
        # Set up for compute device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
    
    def resetParameters(self):
        '''
        Resets the weights of the network.
        *functionName is a "methodcaller" operation in Python
        '''
        for layer in self.NeuralNet:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
    
    def buildNetwork(self):
        '''
        In the DDQN paper, they concatenate the actions into the second layer of the network.
        Ultimately, the actions need to be concatenated somewhere, so first or second layer will
        do the trick. This can be computationally difficult if there is a large action space.
        Problems might also occur if we try to normalize the input and action spaces. If they
        are drastically different, performing the same normilzation on both, then contantenation
        might be undesirable.
        '''
        self.NeuralNet = nn.ModuleList([nn.Linear(self.stateSize, self.hiddenLayers[0])])
        layerSizes = [item for item in zip(self.hiddenLayers[:-1], self.hiddenLayers[1:])]

        
        for idx in range(len(layerSizes)):
            if idx == 0:
                self.NeuralNet.extend([nn.Linear(layerSizes[idx][0] + self.actionSize, layerSizes[idx][1])])      
            else:
                self.NeuralNet.extend([nn.Linear(layerSizes[idx][0], layerSizes[idx][1])])
        self.output = nn.Linear(self.hiddenLayers[-1], 1)
                              
    
    def forward(self,state, action):
        for layerIdx in range(len(self.NeuralNet)):
            if layerIdx == 0:
                x = F.relu(self.NeuralNet[layerIdx](state))
            else:
                if layerIdx == 1:
                    x = torch.cat((x, action), dim=1)
                x = F.relu(self.NeuralNet[layerIdx](x))
        return self.output(x)

class TD3Critic(nn.Module):
    '''
    Special Critic network for the TD3 algorithm. It has two
    separate yet architecturally identical networks called Twins.
    The losses of both twins are combined and used to optimize both
    networks.
    '''
    
    def __init__(self, state_size, action_size, seed, hiddenArray):
        super(TD3Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.stateSize = state_size
        self.actionSize = action_size
        self.hiddenLayers = hiddenArray
        self.NeuralNetA = None
        self.NeuralNetB = None
        self.outputA = None
        self.outputB = None
        self.buildNetworks()
        self.resetParameters()
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
    
    def buildNetworks(self):
        self.NeuralNetA = nn.ModuleList([nn.Linear(self.stateSize + \
                                                   self.actionSize, self.hiddenLayers[0])])
        self.NeuralNetB = nn.ModuleList([nn.Linear(self.stateSize + \
                                                   self.actionSize, self.hiddenLayers[0])])
        
        for i in range(len(self.hiddenLayers) - 1):
            self.NeuralNetA.extend([nn.Linear(self.hiddenLayers[i], self.hiddenLayers[i + 1])])
            self.NeuralNetB.extend([nn.Linear(self.hiddenLayers[i], self.hiddenLayers[i + 1])])
        
        self.outputA = nn.Linear(self.hiddenLayers[-1], 1)
        self.outputB = nn.Linear(self.hiddenLayers[-1], 1)
    
    def resetParameters(self):
        '''
        Resets the weights of the network.
        *functionName is a "methodcaller" operation in Python
        '''
        for layer in self.NeuralNetA:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.outputA.weight.data.uniform_(-3e-3, 3e-3)
        
        for layer in self.NeuralNetB:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.outputB.weight.data.uniform_(-3e-3, 3e-3)
    
    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)
        return x, u
    
    def forward(self, state, action):
        x_a = torch.cat((state, action), dim=1)
        x_b = torch.cat((state, action), dim=1)
        
        for layerIdx_a in range(len(self.NeuralNetA)):
                x_a = F.relu(self.NeuralNetA[layerIdx_a](x_a))
        
        for layerIdx_b in range(len(self.NeuralNetB)):
                x_b = F.relu(self.NeuralNetB[layerIdx_b](x_b))
        
        x_a = self.outputA(x_a)
        x_b = self.outputB(x_b)
        
        return x_a, x_b
    
    def forwardNetA(self, state, action):
        '''
        Used to perform a forward pass only through the A network. This is
        useful for getting target Q-values for Actor (policy) updates
        '''
        x, u = self._format(state, action)
        x_a = torch.cat((x, u), dim=1)
        
        for layerIdx_a in range(len(self.NeuralNetA)):
                x_a = F.relu(self.NeuralNetA[layerIdx_a](x_a))
        
        x_a = self.outputA(x_a)
        
        return x_a
                                         
    