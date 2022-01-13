
import numpy as np
import random
import copy
from collections import deque, namedtuple
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree

from networkModel import Actor, Critic, TD3Critic

import torch
#import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




# Global Parameters
""" Alpha, or Learning Rate """
LEARN_RATE_ACTOR = 1e-4
LEARN_RATE_CRITIC = 1e-3
""" Discount factor for rewards """
GAMMA = 0.99
""" Size of Replay Buffer """
BUFFER_SIZE = int(1e5)
""" Size of batches to learn from the Replay Buffer """
BATCH_SIZE = 4 # was 128
""" Parameter to control how many steps between updating the Neural Net """
LEARN_EVERY = 4
""" Parameters for controlling updates in TD3 """
UPDATE_CRITIC_TARGET_STEPS = 5 # was 2
UPDATE_ACTOR_TARGET_STEPS = 5 # was 2
TRAIN_ACTOR_STEPS = 5 # was 2
""" Parameter to control network soft updates"""
TAU = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    '''
    Structure that contains the mechanics for sampling and learning from the environment
    
    Methods
    step: take the next step in the environment after choosing an action
    act: take an action based on the current state
    learn: apply the rewards gained to the Neural Net
    '''
    
    def __init__(self, actionBoundsArray, stateSize, actionSize, random_seed, fileName=None):
        
        # Size of the state space
        self.stateSize = stateSize
        
        # Size of the action space
        self.actionSize = actionSize
        
        # Get bounds of action space for networks
        self.actionBounds = actionBoundsArray
        
        # PRNG seed
        self.seed = random.seed(random_seed)
        
        # Varaible to keep track of steps in between each NN learning Update
        self.stepNum = 0

        
        # Establish the Target and the Local Q-Networks for Fixed Q-Targets
        # use .to(device) to specify the data type of the torch Tensor (for CPU or GPU)
        
        # Actor Networks
        self.QNet_Actor_Local = Actor(self.actionBounds, self.stateSize, self.actionSize, seed=random_seed, hiddenArray = [400, 300]).to(device)
        self.QNet_Actor_Target = Actor(self.actionBounds, self.stateSize, self.actionSize, seed=random_seed, hiddenArray = [400, 300]).to(device)
        
        # Critic Networks
        self.QNet_Critic_Local = Critic(self.stateSize, self.actionSize, seed=random_seed, hiddenArray = [400, 300]).to(device)
        self.QNet_Critic_Target = Critic(self.stateSize, self.actionSize, seed=random_seed, hiddenArray = [400, 300]).to(device)
        
        # Used to run the network for inference
        if fileName:
            weights = torch.load(filename)
            self.QNet_Actor_Local.load_state_dict(weights)
            self.QNet_Actor_Target.load_state_dict(weights)
        
        # Set up the Optimizer for training the networks. Can set learning rate (i.e. gradient step size) 
        # and apply momentum if desired
        self.Optim_Actor = optim.Adam(self.QNet_Actor_Local.parameters(), lr=LEARN_RATE_ACTOR)
        self.Optim_Critic = optim.Adam(self.QNet_Critic_Local.parameters(), lr=LEARN_RATE_CRITIC)
        
        # Noise process (using Ornstein-Uhlenbeck process)
        # Used to encourage exploration so the Agent doesn't
        # pick greedy actions every step
        self.noise = OUNoise(self.actionSize, random_seed)
        
        
        # Establish the Replay Buffer
        self.replayMem = ReplayBuffer(random_seed)
        
    
    def step(self, state, action, reward, next_state, done):
        # Add step to the Replay Buffer
        self.replayMem.addMem(state, action, reward, next_state, done)
        
        # After LEARN_EVERY number of steps, update the NN
        self.stepNum = (self.stepNum + 1) % LEARN_EVERY
        
        if self.stepNum == 0:
            # Perform learning Action if there are enough samples to make up a batch (set by BATCH_SIZE)
            if len(self.replayMem) > BATCH_SIZE:
                experienceSample = self.replayMem.memSample()
                self.learn(experienceSample)
    
    def act(self, state, addNoise = True):
        '''
        Returns the best guess of action values based on the current state of the NN
        '''
        # Convert state to PyTorch tensor, make sure all values are a float
        # .unsqueeze will reformat to dimensions with 1 at the specified dim (0 - row, 1 - column)
        # .to(device) converts the tensor to the proper type for the device (CPU or GPU)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Turn off Dropout Layers, Batch layers, etc. so we can run the network straight-through/without
        # training mechanisms applied. Called eval mode
        self.QNet_Actor_Local.eval()
        
        with torch.no_grad():
            actionVals = self.QNet_Actor_Local(state).cpu().data.numpy()
        
        # Turn training mode back on
        self.QNet_Actor_Local.train()
        
        if addNoise:
            actionVals += self.noise.sample()
        return np.clip(actionVals, -1, 1)
        
    def learn(self, experiences):
        '''
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        '''

        states, actions, rewards, nextStates, dones = experiences
        
        
        # -------------------------- Update Critic ---------------------- #
        
        # Get predicted next actions from the Actor Target and Q-values from the Critic Target model
        actionsNext = self.QNet_Actor_Target(nextStates)
        Q_Targets_Next = self.QNet_Critic_Target(nextStates, actionsNext)
        
        # Compute the Q-targets from for the current states
        # rewards = R + (Gamma * maximizing action reward * (1-dones)
        Q_Targets = rewards + (GAMMA * Q_Targets_Next * (1 - dones))
        
        # Get the expected Q-values from the Critic Local model using the current states
        Q_Expected = self.QNet_Critic_Local(states, actions)
        
        # Calculate the loss
        criticLoss = F.mse_loss(Q_Expected, Q_Targets)
        
        # Minimize the loss
        self.Optim_Critic.zero_grad()
        criticLoss.backward()
        self.Optim_Critic.step()
        
        # -------------------------- Update Actor ----------------------- #
        # Compute Actor loss
        actionsPredict = self.QNet_Actor_Local(states)
        actorLoss = - self.QNet_Critic_Local(states, actionsPredict).mean()
        
        # Minimize the loss
        self.Optim_Actor.zero_grad()
        actorLoss.backward()
        self.Optim_Actor.step()
        
        # ---------------------- Update Target Networks ------------------ #
        self.softUpdate(self.QNet_Critic_Target, self.QNet_Critic_Local)
        self.softUpdate(self.QNet_Actor_Target, self.QNet_Actor_Local)
        
    def softUpdate(self, targetModel, localModel):
        """
        Soft-update equation, called Polyak update
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Copies the parameters of the local network into the target network in small percentages
        """
        
        for targetParams, localParams in zip(targetModel.parameters(), localModel.parameters()):
            targetParams.data.copy_(TAU * localParams.data + (1.0 - TAU) * targetParams.data)

""" Parameter to control importance weights on replay probabilities
    0 -> no importance, 1 -> full importance
    will be gradually increased from BETA_INITIAL to BETA_FINAL
"""
BETA_INITIAL = 0.4
BETA_FINAL = 1.0
""" Small constant to add to PER buffer so probabilities aren't 0"""
PER_EPS = 1e-6
class TD3Agent():
    '''
    Structure that contains the mechanics for sampling and learning from the environment
    
    Methods
    step: take the next step in the environment after choosing an action
    act: take an action based on the current state
    learn: apply the rewards gained to the Neural Net
    '''
    
    def __init__(self, actionBoundsArray, stateSize, actionSize, random_seed, totalTimeSteps, usePER = False, fileName=None):
        
        # Size of the state space
        self.stateSize = stateSize
        
        # Size of the action space
        self.actionSize = actionSize
        
        # Get bounds of action space for networks
        self.actionBounds = actionBoundsArray
        
        # Variable to choose between normal Replay Buffer and Prioritized
        # Replay Buffer (Prioritized Experience Replay)
        self.shouldUsePER = usePER
        self.betaSchedule = None
        self.tdError = None
        
        # Noise ratio parameters for noisy actions
        self.noiseRatio = 0.1
        self.noiseClipRatio = 0.5
        
        # PRNG seed
        self.seed = random.seed(random_seed)
        
        # Varaible to keep track of steps in between each NN learning Update
        self.stepNum = 0
        
        
        # Establish the Target and the Local Q-Networks for Fixed Q-Targets
        # use .to(device) to specify the data type of the torch Tensor (for CPU or GPU)
        
        # Actor Networks
        self.QNet_Actor_Local = Actor(self.actionBounds, self.stateSize, self.actionSize, seed=random_seed, hiddenArray = [400, 300]).to(device)
        self.QNet_Actor_Target = Actor(self.actionBounds, self.stateSize, self.actionSize, seed=random_seed, hiddenArray = [400, 300]).to(device)
        
        # Critic Networks
        self.QNet_Critic_Local = TD3Critic(self.stateSize, self.actionSize, seed=random_seed, hiddenArray = [400, 300]).to(device)
        self.QNet_Critic_Target = TD3Critic(self.stateSize, self.actionSize, seed=random_seed, hiddenArray = [400, 300]).to(device)
        
        # Used to run the network for inference
        if fileName:
            weights = torch.load(fileName)
            self.QNet_Actor_Local.load_state_dict(weights)
            self.QNet_Actor_Target.load_state_dict(weights)
        
        # Set up the Optimizer for training the networks. Can set learning rate (i.e. gradient step size) 
        # and apply momentum if desired
        self.Optim_Actor = optim.Adam(self.QNet_Actor_Local.parameters(), lr=LEARN_RATE_ACTOR)
        self.Optim_Critic = optim.Adam(self.QNet_Critic_Local.parameters(), lr=LEARN_RATE_CRITIC)
        
        # Noise process (using Ornstein-Uhlenbeck process)
        # Used to encourage exploration so the Agent doesn't
        # pick greedy actions every step
        self.noise = OUNoise(self.actionSize, random_seed)
        
        
        # Establish the Replay Buffer
        if self.shouldUsePER:
            self.betaSchedule = np.linspace(BETA_INITIAL, BETA_FINAL, totalTimeSteps + 1)
            self.replayMem = PrioritizedReplayBuffer(random_seed, alpha=0.6)
        else:
            self.replayMem = ReplayBuffer(random_seed)
        
        # Choose which object will be used for choosing actions
        self.actionStrategy = NormalNoiseDecayStrategy(self.actionBounds)
        
    
    def step(self, state, action, reward, next_state, done):
        # Add step to the Replay Buffer
        self.replayMem.addMem(state, action, reward, next_state, done)
        # After LEARN_EVERY number of steps, update the NN
        self.stepNum += 1
        
        if self.stepNum % LEARN_EVERY == 0:
            # Perform learning Action if there are enough samples to make up a batch (set by BATCH_SIZE)
            if len(self.replayMem) > BATCH_SIZE:
                if self.shouldUsePER:
                    # Calculate TD Error between QNet_Local and QNet_Target for a sampled
                    # experience
                    experienceSample = self.replayMem.memSample(self.betaSchedule[self.stepNum])
                else:
                    experienceSample = self.replayMem.memSample()
                self.learn(experienceSample)
    
    def act(self, state, addNoise = True):
        '''
        Returns the best guess of action values based on the current state of the NN
        '''
        # Convert state to PyTorch tensor, make sure all values are a float
        # .unsqueeze will reformat to dimensions with 1 at the specified dim (0 - row, 1 - column)
        # .to(device) converts the tensor to the proper type for the device (CPU or GPU)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Determine if agent should explore or exploit more often, based if there are a 
        # sufficient number of samples in the Replay Buffer
        minSamples = BUFFER_SIZE * 5
        shouldUseMaxExploration = len(self.replayMem) < minSamples
        actionVals = self.actionStrategy.selectAction(self.QNet_Actor_Local, \
                     state, maxExploration = shouldUseMaxExploration, chooseGreedyAction = False)
         
        return np.clip(actionVals, -1, 1)
        
    def learn(self, experiences):
        '''
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        '''
        if self.shouldUsePER:
            states, actions, rewards, nextStates, dones, weights, batchIdx = experiences
        else:
            states, actions, rewards, nextStates, dones = experiences
            weights, batchIdx = np.ones_like(rewards), None
        
        
        # -------------------------- Update Critic ---------------------- #
        with torch.no_grad():   
            # Get predicted next actions from the Actor Target and Q-values from the Critic Target model
            actionsNext = self.QNet_Actor_Target(nextStates)
            noisyActionsNext = self.actionNoise(actions, actionsNext)
            
            # Get the minimum predicted value from the twin Critic nets
            Q_Targets_Next_A, Q_Targets_Next_B = self.QNet_Critic_Target(nextStates, noisyActionsNext)
            Q_Targets_Next = torch.min(Q_Targets_Next_A, Q_Targets_Next_B)
        
            # Compute the Q-targets from for the current states
            # rewards = R + (Gamma * maximizing action reward * (1-dones)
            Q_Targets = rewards + (GAMMA * Q_Targets_Next * (1 - dones))
        
        # Get the expected Q-values from the Critic Local model using the current states
        Q_Expected_A, Q_Expected_B = self.QNet_Critic_Local(states, actions)
        
        if self.shouldUsePER:
            tempQ_Target = Q_Targets.clone().detach()
            tempQ_Expected_A = Q_Expected_A.clone().detach()
            tdError = torch.mul(abs(tempQ_Target - tempQ_Expected_A) + PER_EPS, \
                                     torch.from_numpy(weights).type(torch.FloatTensor))
            self.replayMem.updatePriorities(batchIdx, tdError[0])
        
        
        # Calculate the loss
        criticLoss_A = F.mse_loss(Q_Expected_A, Q_Targets)
        criticLoss_B = F.mse_loss(Q_Expected_B, Q_Targets)
        criticLoss = criticLoss_A + criticLoss_B
        
        # Minimize the loss
        self.Optim_Critic.zero_grad()
        criticLoss.backward()
        
        # Clip the gradient return just to be safe...
        torch.nn.utils.clip_grad_norm_(self.QNet_Critic_Local.parameters(), float('inf'))
        self.Optim_Critic.step()
        
        # -------------------------- Update Actor ----------------------- #
        # Introduce a delay between updating Critic and Actor networks
        if self.stepNum % TRAIN_ACTOR_STEPS == 0:
            # Compute Actor loss
            actionsPredict = self.QNet_Actor_Local(states)
            actorLoss = -self.QNet_Critic_Local.forwardNetA(states, actionsPredict).mean()
        
            # Minimize the loss
            self.Optim_Actor.zero_grad()
            actorLoss.backward()
            self.Optim_Actor.step()
        
        # ---------------------- Update Target Networks ------------------ #
        # Introduce a delay between updating Critic and Actor Target networks
        if self.stepNum % UPDATE_CRITIC_TARGET_STEPS == 0:
            self.softUpdate(self.QNet_Critic_Target, self.QNet_Critic_Local)
        if self.stepNum % UPDATE_ACTOR_TARGET_STEPS == 0:
            self.softUpdate(self.QNet_Actor_Target, self.QNet_Actor_Local)
        
    def softUpdate(self, targetModel, localModel):
        """
        Soft-update equation, called Polyak update
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Copies the parameters of the local network into the target network in small percentages
        """
        
        for targetParams, localParams in zip(targetModel.parameters(), localModel.parameters()):
            targetParams.data.copy_(TAU * localParams.data + (1.0 - TAU) * targetParams.data)
    
    def actionNoise(self, prevActions, nextActions):
        actionMin = self.actionBounds[0]
        actionMax = self.actionBounds[1]
        actionMinArray = (torch.from_numpy(np.ones(self.actionSize) * \
                                           actionMin)).type(torch.FloatTensor)
        actionMaxArray = (torch.from_numpy(np.ones(self.actionSize) * \
                                           actionMax)).type(torch.FloatTensor)
        
        actionRange = actionMaxArray - actionMinArray
        actionNoise = torch.randn_like(prevActions)  * self.noiseRatio * actionRange
        noiseMin = actionMinArray * self.noiseClipRatio
        noiseMax = actionMaxArray * self.noiseClipRatio
        actionNoise = torch.max(torch.min(actionNoise, noiseMax), noiseMin)
        
        noisyNextActions = nextActions + actionNoise
        noisyAction = torch.max(torch.min(noisyNextActions, actionMaxArray), actionMinArray)
        
        return noisyAction
        
class OUNoise:
    '''Ornstein-Uhlenbeck Noise Process '''
    
    def __init__(self, size, seed, mu = 0.0, theta = 0.15, sigma = 0.2):
        ''' 
        Initialization parameters taken from CONTINUOUS CONTROL WITH DEEP REINFORCEMENT
        LEARNING paper
        Sampling the OU process can be described with an ODE
        dot_x = theta * (mu - x) + sigma
        '''
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.seed = random.seed(seed)
        self.reset()
    
    def reset(self):
        '''
        Reset the internal state to the mean, mu
        Requires package 'copy'
        '''
        self.state = copy.copy(self.mu)
        
    def sample(self):
        '''
        Update the internal state and return it as a noise sample
        '''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class NormalNoiseDecayStrategy():
    def __init__(self, bounds, initialNoiseRatio = 0.5, minNoiseRatio = 0.1, decaySteps = 10000):
        '''
        bounds: usually the action bounds
        '''
        self.step = 0
        self.low = bounds[0]
        self.high = bounds[1]
        self.initNoiseRatio = initialNoiseRatio
        self.minNoiseRatio = minNoiseRatio
        self.decaySteps = decaySteps
        self.ratioNoiseInjected = 0
    
    def noiseRatioUpdate(self):
        noiseRatio = 1 - self.step / self.decaySteps
        
        # Scale the noise ratio
        noiseRatio = (self.initNoiseRatio - self.minNoiseRatio) * noiseRatio + self.minNoiseRatio
        noiseRatio = np.clip(noiseRatio, self.minNoiseRatio, self.initNoiseRatio)
        
        self.step += 1
        
        return noiseRatio
    
    def selectAction(self, network, state, maxExploration = False, chooseGreedyAction = False):
        if maxExploration:
            noiseScale = self.high
        else:
            noiseScale = self.noiseRatio * self.high
        
        network.eval()
        with torch.no_grad():
            greedyAction = network(state).cpu().detach().data.numpy().squeeze()
        network.train()
        
        if chooseGreedyAction:
            self.ratioNoiseInjected = 0
            return np.clip(greedyAction, self.low, self.high)
        
        noise = np.random.normal(loc=0, scale=noiseScale, size=len(greedyAction))
        
        noisyAction = greedyAction + noise
        action = np.clip(noisyAction, self.low, self.high)
        
        self.noiseRatio = self.noiseRatioUpdate()
        
        # Used to inject noise into other steps of the process
        self.ratioNoiseInjected = np.mean(abs((greedyAction - action)/(self.high - self.low)))

        return action

class GreedyStrategy():
    '''
    I think I fixed this by adding a chooseGreedyAction bool to
    the NormalNoiseDecayStrategy
    '''
    def _init__(self, bounds):
        self.low = bounds[0]
        self.high = bounds[1]
        self.ratioNoiseInjected = 0
    
    def selectAction(self, model, state):
        with torch.no_grad():
            greedyAction = model(state).cpu.detach().data.numpy().squeeze()
        action = np.clip(greedyAction, self.low, self.high)
        return action

class ReplayBuffer():
    '''
    Structure to keep track of previous state, action pairs for Experience Replay
    
    Methods
    memAdd: store a memory in the buffer
    memSample: sample a memory from the buffer
    '''
    
    def __init__(self, seed):
        self.memBuff = deque(maxlen = BUFFER_SIZE)
        self.batchSize = BATCH_SIZE
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "nextState", "done"])
        self.seed = random.seed(seed)
        
    def addMem(self, state, action, reward, nextState, done):
        # Store the data in an experience tuple
        e = self.experience(state, action, reward, nextState, done)
        
        # Add the tuple to the buffer
        self.memBuff.append(e)
    
    def memSample(self):
        # Choose a random sample of memory from the Replay Buffer
        memorySample = random.sample(self.memBuff, k=self.batchSize)
        
        # Separate the sample into constituent parts
        states = torch.from_numpy(np.vstack([mem.state for mem in memorySample if mem is not \
                                             None])).float().to(device)
        actions = torch.from_numpy(np.vstack([mem.action for mem in memorySample if mem is not \
                                              None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([mem.reward for mem in memorySample if mem is not \
                                              None])).float().to(device)
        nextStates = torch.from_numpy(np.vstack([mem.nextState for mem in memorySample if mem is not \
                                                 None])).float().to(device)
        dones = torch.from_numpy(np.vstack([mem.done for mem in memorySample if mem is not \
                                            None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, nextStates, dones)

    def __len__(self):
        # Convenience function for returning the length of the buffer
        return len(self.memBuff)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, seed, alpha=0.5):
        super(PrioritizedReplayBuffer, self).__init__(seed)
        self.alpha = alpha
        capacity = 1
        
        while capacity < BUFFER_SIZE:
            capacity *= 2
        
        self.sumTree = SumSegmentTree(capacity)
        self.minTree = MinSegmentTree(capacity)
        self.maxPriority = 1.0
        self.nextIdx = 0
        
    def addMem(self, *args, **kwargs):
        idx = self.nextIdx
        super().addMem(*args, **kwargs)
        self.sumTree[idx] = self.maxPriority ** self.alpha
        self.minTree[idx] = self.maxPriority ** self.alpha
        
        self.nextIdx = (self.nextIdx + 1) % BUFFER_SIZE
   
    def sampleProportional(self):
        result = []
        probTotal = self.sumTree.sum(0, len(self.memBuff) - 1)
        everyRangeLen = probTotal/BATCH_SIZE
        
        for i in range(BATCH_SIZE):
            mass = random.random() * everyRangeLen + i * everyRangeLen
            idx = self.sumTree.find_prefixsum_idx(mass)
            result.append(idx)
        return result
    
    def encodedSample(self, idxes):
        states = []
        actions = []
        rewards = []
        nextStates = []
        dones = []
        for idx, count in zip(idxes, range(len(idxes))):
            sampleChoice = self.memBuff[idx]
            if sampleChoice is not None:
                states.append(sampleChoice.state)
                actions.append(sampleChoice.action)
                rewards.append(sampleChoice.reward)
                nextStates.append(sampleChoice.nextState)
                dones.append(sampleChoice.done)
                #states[(count,0)] = sampleChoice.state
                #actions[count] = sampleChoice.action
                #rewards[count] = sampleChoice.reward
                #nextStates[count] = sampleChoice.nextState
                #dones[count] = sampleChoice.done
                
        #states = torch.tensor(states).float().to(device)
        #actions = torch.tensor(actions).float().to(device)
        #rewards = torch.tensor(rewards).float().to(device)
        #nextStates = torch.tensor(nextStates).float().to(device)
        #dones = torch.tensor(dones).float().to(device)
        
        # Need to use unsqueeze because need an extra 1 in the dimensions for rewards and dones
        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.from_numpy(np.array(actions)).float().to(device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device).unsqueeze(1)
        nextStates = torch.from_numpy(np.array(nextStates)).float().to(device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(device).unsqueeze(1)
        
        return (states, actions, rewards, nextStates, dones)
    
    def memSample(self, beta=0.5):
        indexes = self.sampleProportional()
        
        weights = []
        probMin = self.minTree.min() / self.sumTree.sum()
        maxWeight = (probMin * len(self.memBuff)) ** (-beta)
        
        for idx in indexes:
            probSample = self.sumTree[idx] / self.sumTree.sum()
            weight = (probSample * len(self.memBuff)) ** (-beta)
            weights.append(weight / maxWeight)
        weights = np.array(weights)
        encodedSampleReturn = self.encodedSample(indexes)
        return tuple(list(encodedSampleReturn) + [weights, indexes])
    
    def updatePriorities(self, indexes, priorities):
        assert len(indexes) == len(priorities)

        for idx, priority in zip(indexes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.memBuff)
            self.sumTree[idx] = priority ** self.alpha
            self.minTree[idx] = priority ** self.alpha
            
            self.maxPriority = max(self.maxPriority, priority)