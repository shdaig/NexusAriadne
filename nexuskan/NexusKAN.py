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
        
        self.input_id = torch.arange(self.width_in[0],)
    
    def to(self, device):
        super(NexusKAN, self).to(device)
        self.device = device
        for kanlayer in self.act_fun:
            kanlayer.to(device)
        return self
    
    @property
    def width_in(self):
        '''
        The number of input nodes for each layer
        '''
        width = self.width
        width_in = [width[l][0] + width[l][1] for l in range(len(width))]
        return width_in
    
    @property
    def width_out(self):
        '''
        The number of output subnodes for each layer
        '''
        # ---------------------------------------------------------------------------------------------------------------------------------
        width = self.width
        if self.mult_homo == True:
            width_out = [width[l][0]+self.mult_arity*width[l][1] for l in range(len(width))]
        else:
            width_out = [width[l][0] + int(np.sum(self.mult_arity[l])) for l in range(len(width))]
        return width_out
    
    @property
    def n_sum(self):
        '''
        The number of addition nodes for each layer
        '''
        width = self.width
        n_sum = [width[l][0] for l in range(1, len(width)-1)]
        return n_sum
    
    @property
    def n_mult(self):
        # ---------------------------------------------------------------------------------------------------------------------------------
        '''
        The number of multiplication nodes for each layer
        '''
        width = self.width
        n_mult = [width[l][1] for l in range(1, len(width)-1)]
        return n_mult
    
    def update_grid_from_samples(self, x):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.tensor
                inputs

        Returns:
        --------
            None
            
        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
        >>> print(model.act_fun[0].grid)
        >>> x = torch.linspace(-10,10,steps=101)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.act_fun[0].grid)
        ''' 
        for l in range(self.depth):
            self.get_act(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])
    
    def update_grid(self, x):
        '''
        call update_grid_from_samples. This seems unnecessary but we retain it for the sake of classes that might inherit from MultKAN
        '''
        self.update_grid_from_samples(x)
        
    def forward(self, x, singularity_avoiding=False, y_th=10.):
       
        x = x[:,self.input_id.long()]
        assert x.shape[1] == self.width_in[0]
        
        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        
        self.acts_premult = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_spline = []
        self.subnode_actscale = []
        self.edge_actscale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        for l in range(self.depth):
            
            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)
            
            x = x_numerical
            
            # subnode affine transform
            x = self.subnode_scale[l][None,:] * x + self.subnode_bias[l][None,:]
            
            # multiplication
            dim_sum = self.width[l+1][0]
            dim_mult = self.width[l+1][1]
            
            if self.mult_homo == True:
                for i in range(self.mult_arity-1):
                    if i == 0:
                        x_mult = x[:,dim_sum::self.mult_arity] * x[:,dim_sum+1::self.mult_arity]
                    else:
                        x_mult = x_mult * x[:,dim_sum+i+1::self.mult_arity]
                        
            else:
                for j in range(dim_mult):
                    acml_id = dim_sum + np.sum(self.mult_arity[l+1][:j])
                    for i in range(self.mult_arity[l+1][j]-1):
                        if i == 0:
                            x_mult_j = x[:,[acml_id]] * x[:,[acml_id+1]]
                        else:
                            x_mult_j = x_mult_j * x[:,[acml_id+i+1]]
                            
                    if j == 0:
                        x_mult = x_mult_j
                    else:
                        x_mult = torch.cat([x_mult, x_mult_j], dim=1)
                
            if self.width[l+1][1] > 0:
                x = torch.cat([x[:,:dim_sum], x_mult], dim=1)
            
            # x = x + self.biases[l].weight
            # node affine transform
            x = self.node_scale[l][None,:] * x + self.node_bias[l][None,:]
            
            self.acts.append(x.detach())
            
        
        return x