# -*- coding: utf-8 -*-
"""

"""

import dill
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCRNN(nn.Module):
    
    def __init__(self, config):
        
        super(BaseCRNN, self).__init__()
        
        self.config = config
        self.len_data = config['len_data']
        self.len_split = config['len_split']
        self.n_seg = int(self.len_data/self.len_split)
        self.input_channel = config['n_channel']
        self.n_class = config['n_class']
        self.is_debug = config['is_debug']
        self.use_conv2 = config['use_conv2']
        
        ### conv config
        self.conv1 = nn.Conv1d(in_channels=self.input_channel, 
                                out_channels=config['conv']['filters'], 
                                kernel_size=config['conv']['kernel_size'], 
                                stride=config['conv']['strides'])
        self.do1 = nn.Dropout(p=0.5)
                
        self.conv2 = nn.Conv1d(in_channels=config['conv']['filters'], 
                                out_channels=2*config['conv']['filters'], 
                                kernel_size=config['conv']['kernel_size'], 
                                stride=config['conv']['strides'])
        self.do2 = nn.Dropout(p=0.5)
        
        ### Input: (batch size, length of signal sequence, input_size)
        if self.use_conv2:
            self.rnn = nn.LSTM(input_size=2*config['conv']['filters'], 
                                hidden_size=config['rnn']['hidden_size'], 
                                num_layers=1, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_size=config['conv']['filters'], 
                                hidden_size=config['rnn']['hidden_size'], 
                                num_layers=1, batch_first=True, bidirectional=True)
        
        self.W_att = nn.Parameter(torch.randn(2*config['rnn']['hidden_size'], 1))
        
        ### dense
        self.dense = nn.Linear(2*config['rnn']['hidden_size'], self.n_class)
        
        ### trainable weight
        self.w1 = nn.Parameter(torch.Tensor([0.5]))
        self.w2 = nn.Parameter(torch.Tensor([0.5]))
        self.b = nn.Parameter(torch.Tensor([0.0]))

        # init
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)        
        nn.init.xavier_uniform_(self.W_att)
        nn.init.xavier_uniform_(self.dense.weight)
    
    def forward(self, x, temperature):
        
        self.batch_size = x.size()[0]

        ### reshape
        if self.is_debug:
            print('orignial x:', x.size())
        x = x.view(-1, self.len_split, self.input_channel)
        if self.is_debug:
            print('reshaped for cnn:', x.size())
            
        ### conv
        x = x.permute(0, 2, 1)
        if self.is_debug:
            print('before conv1:', x.size())
        x = F.relu(self.conv1(x))
        if self.is_debug:
            print('after conv1:', x.size())
        x = self.do1(x)
                
        if self.use_conv2:
            ### conv
            if self.is_debug:
                print('before conv2:', x.size())
            x = F.relu(self.conv2(x))
            if self.is_debug:
                print('after conv2:', x.size())
            x = self.do2(x)

        ### pooling
        x = torch.mean(x, dim=-1)
        if self.is_debug:
            print('after pooling in conv:', x.size())
        
        ### reshape for rnn
        x = x.view(self.batch_size, self.n_seg, -1)
        if self.is_debug:
            print('reshaped for rnn:', x.size())
            
        ### rnn        
        o, (ht, ct) = self.rnn(x)
        if self.is_debug:
            print('outs in rnn', o.size(), ht.size(), ct.size())
        e = torch.matmul(o, self.W_att)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        att_weights = torch.div(n1, n2)
        if self.is_debug:
            print('att_weights', att_weights.size())
        x = torch.sum(torch.mul(att_weights, o), 1)
        if self.is_debug:
            print('after att in rnn:', x.size())
            
        att_weights = torch.squeeze(att_weights)
        
        ### dense
        x = self.dense(x)
        
        ### predicts
        x_raw = F.softmax(x, dim=-1)
        x_temp = F.softmax(torch.div(x, temperature), dim=-1)

        return x_raw, x_temp, att_weights


    
class BaseCNN(nn.Module):
    
    def __init__(self, config):
        
        super(BaseCNN, self).__init__()
        
        self.config = config
        self.input_channel = config['n_channel']
        self.n_class = config['n_class']
        self.is_debug = config['is_debug']
        
        ### conv config
        self.conv1 = nn.Conv1d(in_channels=self.input_channel, 
                                out_channels=config['conv']['filters'], 
                                kernel_size=config['conv']['kernel_size'], 
                                stride=config['conv']['strides'])
        self.do1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv1d(in_channels=config['conv']['filters'], 
                                out_channels=2*config['conv']['filters'], 
                                kernel_size=config['conv']['kernel_size'], 
                                stride=config['conv']['strides'])
        self.do2 = nn.Dropout(p=0.5)

        ### dense
        self.dense = nn.Linear(2*config['conv']['filters'], self.n_class)
        
        # init
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant(self.conv1.bias, 0)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant(self.conv2.bias, 0)        
        nn.init.xavier_uniform_(self.dense.weight)
    
    def forward(self, x, temperature):
        
        ### conv
        x = x.permute(0, 2, 1)
        if self.is_debug:
            print('before conv1:', x.size())
        x = F.relu(self.conv1(x))
        if self.is_debug:
            print('after conv1:', x.size())
        x = self.do1(x)
                
        ### conv
        if self.is_debug:
            print('before conv2:', x.size())
        x = F.relu(self.conv2(x))
        if self.is_debug:
            print('after conv2:', x.size())
        x = self.do2(x)

        ### pooling
        x = torch.mean(x, dim=-1)
        if self.is_debug:
            print('after pooling in conv:', x.size())
        
        ### dense
        x = self.dense(x)
        
        ### predicts
        x_raw = F.softmax(x, dim=-1)
        x_temp = F.softmax(torch.div(x, temperature), dim=-1)

        return x_raw, x_temp, x_temp


    