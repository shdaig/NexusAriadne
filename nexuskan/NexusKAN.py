import torch
import torch.nn as nn

import numpy as np

import random

from .NexusKANLayer import NexusKANLayer

class NexusKAN(nn.Module):
    def __init__(self, 
                 width=None, 
                 grid=3, 
                 k=3, 
                #  mult_arity = 2, 
                 noise_scale=0.3, 
                 scale_base_mu=0.0, 
                 scale_base_sigma=1.0, 
                 base_fun='silu', 
                 symbolic_enabled=True, 
                 affine_trainable=False, 
                 grid_eps=0.02, 
                 grid_range=[-1, 1],
                 sp_trainable=True, 
                 sb_trainable=True, 
                 seed=1, 
                 save_act=True, 
                 sparse_init=False, 
                #  auto_save=True, 
                 first_init=True, 
                 ckpt_path='./model', 
                 state_id=0, 
                 round=0, 
                 device='cpu'):
        
        super(NexusKAN, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ### initializeing the numerical front ###

        # self.act_fun = []
        self.depth = len(width) - 1
        
        for i in range(len(width)):
            if type(width[i]) == int or type(width[i]) == np.int64:
                width[i] = [width[i],0]
                
        self.width = width
        
        # if mult_arity is just a scalar, we extend it to a list of lists
        # e.g, mult_arity = [[2,3],[4]] means that in the first hidden layer, 2 mult ops have arity 2 and 3, respectively;
        # in the second hidden layer, 1 mult op has arity 4.
        # if isinstance(mult_arity, int):
            # self.mult_homo = True # when homo is True, parallelization is possible
        # else:
            # self.mult_homo = False # when home if False, for loop is required. 
        # self.mult_arity = mult_arity

        width_in = self.width_in
        width_out = self.width_out
        
        self.base_fun_name = base_fun
        if base_fun == 'silu':
            base_fun = torch.nn.SiLU()
        elif base_fun == 'identity':
            base_fun = torch.nn.Identity()
        elif base_fun == 'zero':
            base_fun = lambda x: x*0.
            
        self.grid_eps = grid_eps
        self.grid_range = grid_range
            
        for l in range(self.depth):
            # splines
            if isinstance(grid, list):
                grid_l = grid[l]
            else:
                grid_l = grid
                
            if isinstance(k, list):
                k_l = k[l]
            else:
                k_l = k
            
            sp_batch = NexusKANLayer(in_dim=width_in[l], 
                                     out_dim=width_out[l+1], 
                                     num=grid_l, 
                                     k=k_l, 
                                     noise_scale=noise_scale, 
                                     scale_base_mu=scale_base_mu, 
                                     scale_base_sigma=scale_base_sigma, 
                                     scale_sp=1., 
                                     base_fun=base_fun, 
                                     grid_eps=grid_eps, 
                                     grid_range=grid_range, 
                                     sp_trainable=sp_trainable, 
                                     sb_trainable=sb_trainable, 
                                     sparse_init=sparse_init)
            self.act_fun.append(sp_batch)
        
        self.affine_trainable = affine_trainable
        
        self.node_bias = []
        self.node_scale = []
        self.subnode_bias = []
        self.subnode_scale = []
        
        for l in range(self.depth):
            exec(f'self.node_bias_{l} = torch.nn.Parameter(torch.zeros(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_scale_{l} = torch.nn.Parameter(torch.ones(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_bias_{l} = torch.nn.Parameter(torch.zeros(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_scale_{l} = torch.nn.Parameter(torch.ones(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_bias.append(self.node_bias_{l})')
            exec(f'self.node_scale.append(self.node_scale_{l})')
            exec(f'self.subnode_bias.append(self.subnode_bias_{l})')
            exec(f'self.subnode_scale.append(self.subnode_scale_{l})')
            
        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun
        
        self.device = device
        self.to(device)
        
        
        
        

        