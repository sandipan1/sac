import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(ValueNetwork,self).__init__()

        self.layer1 = nn.Linear (state_dim,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.layer3 = nn.Linear(hidden_dim,1)

    def forward(self,state):
        y = F.relu(self.layer1(state))
        y = F.relu(self.layer2(x))
        y = F.layer3(y)

        return y



class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size):
        super(QNetwork,self).__init__()


    ## Q1 Network
        self.layer1 = nn.Linear (state_dim + action_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim,1)

    ## Q2 Network
        self.layer1_ = nn.Linear (state_dim + action_dim, hidden_dim)
        self.layer2_ = nn.Linear(hidden_dim, hidden_dim)
        self.layer3_ = nn.Linear(hidden_dim,1)


    def forward (self, state, action):
        x1 = torch.cat([state,action],1)

        # Q1 Network
        y1 = F.relu(self.layer1(x1))
        y1 = F.relu(self.layer2(y1))
        y1 = F.layer3(y1)

        # Q2 Network
        y1_ = F.relu(self.layer1_(x1))
        y1_ = F.relu(self.layer2_(y1))
        y1_ = F.layer3_(y1_)

        return y1,y1_




