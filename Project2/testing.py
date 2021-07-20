import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import Categorical


Gamma = 0.99
seed_value = 57899
learning_rate=.003




env = gym.make("LunarLander-v2")

env.seed(seed_value)
torch.manual_seed(seed_value)


class Policy_Net(nn.Module):
    def __init__(self):
        super(Policy_Net,self).__init__()
        self.input_layer = nn.Linear(env.observation_space.shape[0],28)
        
        self.action_layer = nn.Linear(28,env.action_space.n)
        self.Value_layer = nn.Linear(28,1)
        
        self.saved_log_probs = []
        self.state_val = []
        self.reward_vec = []
        
    def forward(self,state):
        state = self.input_layer(state)
        state = F.dropout(state, p =.3)
        state = F.relu(state)
        
        state_value = self.Value_layer(state)

     
        self.prob_d = F.softmax(self.action_layer(state), dim =1)
        self.m = Categorical(self.prob_d)
        self.action = self.m.sample()
        
        ## m.log_prob takes the natural log of the probability associated with action
        self.saved_log_probs.append(self.m.log_prob(self.action))
        ## approximate the Value of a given state
        self.state_val.append(state_value)

        

        return self.action

 


policy = Policy_Net()
policy.load_state_dict(torch.load('model'))
policy.eval()
plot_reward = []
counter = 0
while True:
    if counter==100:
        break
    state = env.reset()
    ep_reward = 0
    
    while True:
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        probs = policy(state)
        
        action = probs.item()

        state, reward, done, _ = env.step(action)
        ep_reward+=reward
        #env.render()
        
        if done:
            counter+=1
            print("Reward for episode ", counter,ep_reward)
            plot_reward.append(ep_reward)
            ep_reward=0
            break
plt.plot(plot_reward)
plt.show()
        
