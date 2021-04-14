#-*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorModel(nn.Module):#actor网络
    def __init__(self, obsinput_size, output_size):
        super(ActorModel, self).__init__()
        hid_size1 = 64
        hid_size2 = 128
        self.fc_layer1 = nn.Linear(obsinput_size, hid_size1)#全连接层
        self.fc_layer2 = nn.Linear(hid_size1, hid_size2)
        self.fc_layer3 = nn.Linear(hid_size2, output_size)

    def forward(self, obs):
        #print(obs.shape)
        x = torch.tanh(self.fc_layer1(obs))
        x = torch.tanh(self.fc_layer2(x))
        x = F.softmax(self.fc_layer3(x),dim=1)
        #print(x)
        return x

class CriticModel(nn.Module):#Q网络
    def __init__(self, input_size,output_size):
        super().__init__()
        hid_size1 = 64
        hid_size2 = 128
        self.fc_layer1 = nn.Linear(input_size, hid_size1)#全连接层
        self.fc_layer2 = nn.Linear(hid_size1, hid_size2)
        self.fc_layer3 = nn.Linear(hid_size2, output_size)

    def forward(self, obs, act):
        #print(obs.shape)
        #print(act.shape)
        concat = torch.cat([obs, act], 1)
        hid1 = torch.tanh(self.fc_layer1(concat))
        hid2 = torch.tanh(self.fc_layer2(hid1))
        Q = self.fc_layer3(hid2)
        Q = torch.squeeze(Q, dim=1)
        return Q

