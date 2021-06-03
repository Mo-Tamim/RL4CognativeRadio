# Environemnt Class 
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch 
import gym 
from gym import spaces
from PUs.PU import PU 

class Env(gym.Env): 

    def __init__(self, Horizon=20) -> None:
        super(Env,self).__init__()
        self.action_space = spaces.MultiBinary(1)
        self.observation_space = spaces.Discrete(Horizon + 1 )
        self.Horizon = Horizon
        self.Timer = 0 
        self.createPU(Horizon)
        

    def createPU (self, Horizon):
        PU1= PU(Horizon)
        PU1.createTxPattern()
        self.TxPattern = PU1.TxPattern

    def reset(self):
        self.Timer = 0
        o = torch.tensor([self.Timer], dtype=torch.float)
        return o
    

    def step(self, action, o ): #action is a binary  0 or 1, in case of single PU 
        # assert self.action_space.contains(action[0])
        
        self.Timer = int(o)
        if self.Timer < self.Horizon:  # non-terminal observation, horizon not reached
            r = torch.sum((action == self.TxPattern[self.Timer,:])) 

        else:  # gym horizon reached
            r = 0.0

        self.Timer += 1  # increment our time-step / observation
        o = torch.tensor([self.Timer], dtype=torch.float) # observation that will return
        done = (self.Timer == torch.tensor([self.Horizon]))  # is terminal gym state reached
        # print("o: {} r: {} action: {}".format(o,r,action))
        return o, r, done, {}  # gyms always returns <obs, reward, terminal obs reached, debug/info dictionary>

    def render(self, mode='human'):
        print('Lets see TxPattern \n')
        print(self.TxPattern)


if __name__ == "__main__": 
    TestEnv = Env()
    print(TestEnv.TxPattern)