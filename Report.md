# Project 2: Continuous Navigation Using DDPG and TD3
This project introduces the concept of Continuous Action Spaces, which require careful attention in choosing a compatible agent. Previously used architectures such as DQN/DDQN can only operate in discrete action spaces. Architectures such as Deep Deterministic Policy Gradient (DDPG) and Twin-Delayed DDPG (TD3) are comparable to DQN/DDQN with changes to admit Continuous Action Spaces. In these architectures, the agent learns the policy to optmize the action-value function using an Actor-Critic netowrk. DDPG learns a greedy determinsitic policy, but has an off-policy (i.e. stochastic) exploration strategy. TD3 adds a twin network to improve sample efficiency and reduce training time.

The environment is based on the Unity-ML Reacher environment, with the option to use 1 or 20 agents. I chose to use 1 agent for this project.

Section I discusses the architecture of DDPG. Section II covers the changes required to create the TD3 architecture. Section III reviews the hyperparameters used in both designs, Section IV contains the results from the training, and Section V provides the Conclusion and suggestions for future work.

## I. Deep Deterministic Policy Gradient Architecture
## II. Twin-Delayed DDPG Architecture
## III. Hyperparameters
## IV. Training Results
## V. Conclusions and Future Work
