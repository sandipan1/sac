import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from update import soft_update, hard_update  ## check


from policy import TanhGaussianPolicy
from value import QNetwork, ValueNetwork


class SAC:
    def __init__(self,action_dim,state_dim,**args ):   ## TO do
        self.action_dim= action_dim
        self.state_dim=state_dim
        self.gamma= args['gamma']
        self.scale_reward=args['scale_reward']
        self.reparam = args['reparam']
        self.deterministic= args['deterministic']
        self.target_update_interval = args['target_update_interval']
        self.hidden_dim = args['hidden_dim']
        self.lr=args['lr']

        self.policy = TanhGaussianPolicy(self.state_dim,self.action_dim, self.hidden_dim)
        self.policy_optimizer= Adam(self.policy.parameters(), lr=self.lr)

        self.critic = QNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_optimizer- Adam(self.policy.parameters(),lr=self.lr)

        ## hard and soft updates






