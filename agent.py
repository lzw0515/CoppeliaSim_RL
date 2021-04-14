
#-*- coding: utf-8 -*-

import numpy as np
from model import ActorModel,CriticModel
import torch
import torch.optim as optim
import torch.nn as nn


class Agent(object):
    def __init__(self, **kwargs):
        #复制参数名和值
        for key, value in kwargs.items():
            setattr(self, key, value)

        obs_dim = self.obs_dim
        act_dim = self.act_dim

        self.actor = ActorModel(obs_dim, act_dim)
        self.actor_target = ActorModel(obs_dim, act_dim)
        self.critic = CriticModel(obs_dim+act_dim, 1)
        self.critic_target = CriticModel(obs_dim+act_dim, 1)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def predict(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = self.actor(s0).squeeze(0).detach().numpy()#softmax结果
        a1 = np.zeros(4)
        a1[np.argmax(a0)] = 1

        return a1
    
    def put(self, *transition): 
        if len(self.buffer)== self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def learn(self, obs, act, reward, next_obs, terminal):

        #print("start learn!!")
        
        #samples = random.sample(self.buffer, self.batch_size)
        
        #s0, a0, r1, s1 = zip(*samples)
        
        s0 = torch.tensor(obs, dtype=torch.float)
        a0 = torch.tensor(act, dtype=torch.float)
        r1 = torch.tensor(reward, dtype=torch.float)
        #print(r1.shape)
        #r1 = r1.squeeze(1)
        #r1 = torch.tensor(reward, dtype=torch.float).view(self.batch_size,-1)#拉伸成1行
        s1 = torch.tensor(next_obs, dtype=torch.float)
        
        def critic_learn():#Q网络
            a1 = self.actor_target(s1).detach()#计算next_action   detach就是截断反向传播的梯度流
            #print(a1.shape)
            #print(s1.shape)
            #print("!")
            kkk = self.gamma * self.critic_target(s1, a1).detach()
            y_true = r1 + kkk
            
            y_pred = self.critic(s0, a0)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)

            self.critic_optim.zero_grad()#把梯度置零，也就是把loss关于weight的导数变成0.
            loss.backward()
            self.critic_optim.step()
            
        def actor_learn():
            loss = - torch.mean( self.critic(s0, self.actor(s0)) )
            self.actor_optim.zero_grad() #清空过往梯度
            loss.backward() #反向传播，计算当前梯度；
            self.actor_optim.step() #根据梯度更新网络参数
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
    

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}critic.pkl'.format(output)
        )
