import torch
import torch.nn as nn
import numpy as np
import sympy
from .utils import *



class Symbolic_KANLayer(nn.Module):
    '''
    KANLayer class

    Attributes:
    -----------
        in_dim : int
            input dimension
        out_dim : int
            output dimension
        funs : 2D array of torch functions (or lambda functions)
            symbolic functions (torch)
        funs_avoid_singularity : 2D array of torch functions (or lambda functions) with singularity avoiding
        funs_name : 2D arry of str
            names of symbolic functions
        funs_sympy : 2D array of sympy functions (or lambda functions)
            symbolic functions (sympy)
        affine : 3D array of floats
            affine transformations of inputs and outputs
    '''
    def __init__(self, in_dim=3, out_dim=2, device='cpu'):
        '''
        initialize a Symbolic_KANLayer (activation functions are initialized to be identity functions)
        
        Args:
        -----
            in_dim : int
                input dimension
            out_dim : int
                output dimension
            device : str
                device
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=3)
        >>> len(sb.funs), len(sb.funs[0])
        '''
        super(Symbolic_KANLayer, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mask = torch.nn.Parameter(torch.zeros(out_dim, in_dim, device=device)).requires_grad_(False)
        # torch
        self.funs = [[lambda x: x*0. for i in range(self.in_dim)] for j in range(self.out_dim)]
        self.funs_avoid_singularity = [[lambda x, y_th: ((), x*0.) for i in range(self.in_dim)] for j in range(self.out_dim)]
        # name
        self.funs_name = [['0' for i in range(self.in_dim)] for j in range(self.out_dim)]
        # sympy
        self.funs_sympy = [[lambda x: x*0. for i in range(self.in_dim)] for j in range(self.out_dim)]
        ### make funs_name the only parameter, and make others as the properties of funs_name?
        
        self.affine = torch.nn.Parameter(torch.zeros(out_dim, in_dim, 4, device=device))
        # c*f(a*x+b)+d
        
        self.device = device
        self.to(device)
        
    def to(self, device):
        '''
        move to device
        '''
        super(Symbolic_KANLayer, self).to(device)
        self.device = device    
        return self
    
    def forward(self, x, singularity_avoiding=False, y_th=10.):
        '''
        forward
        
        Args:
        -----
            x : 2D array
                inputs, shape (batch, input dimension)
            singularity_avoiding : bool
                if True, funs_avoid_singularity is used; if False, funs is used. 
            y_th : float
                the singularity threshold
            
        Returns:
        --------
            y : 2D array
                outputs, shape (batch, output dimension)
            postacts : 3D array
                activations after activation functions but before being summed on nodes
        
        Example
        -------
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, postacts = sb(x)
        >>> y.shape, postacts.shape
        (torch.Size([100, 5]), torch.Size([100, 5, 3]))
        '''
        
        batch = x.shape[0]
        postacts = []

        for i in range(self.in_dim):
            postacts_ = []
            for j in range(self.out_dim):
                if singularity_avoiding:
                    xij = self.affine[j,i,2]*self.funs_avoid_singularity[j][i](self.affine[j,i,0]*x[:,[i]]+self.affine[j,i,1], torch.tensor(y_th))[1]+self.affine[j,i,3]
                else:
                    xij = self.affine[j,i,2]*self.funs[j][i](self.affine[j,i,0]*x[:,[i]]+self.affine[j,i,1])+self.affine[j,i,3]
                postacts_.append(self.mask[j][i]*xij)
            postacts.append(torch.stack(postacts_))

        postacts = torch.stack(postacts)
        postacts = postacts.permute(2,1,0,3)[:,:,:,0]
        y = torch.sum(postacts, dim=2)
        
        return y, postacts
        
        


    def fix_symbolic(self, i, j, fun_name, x=None, y=None, random=False, a_range=(-10,10), b_range=(-10,10), verbose=True):
        '''
        fix an activation function to be symbolic
        
        Args:
        -----
            i : int
                the id of input neuron
            j : int 
                the id of output neuron
            fun_name : str
                the name of the symbolic functions
            x : 1D array
                preactivations
            y : 1D array
                postactivations
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of a
            verbose : bool
                print more information if True
            
        Returns:
        --------
            r2 (coefficient of determination)
            
        Example 1
        ---------
        >>> # when x & y are not provided. Affine parameters are set to a = 1, b = 0, c = 1, d = 0
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
        >>> sb.fix_symbolic(2,1,'sin')
        >>> print(sb.funs_name)
        >>> print(sb.affine)
        
        Example 2
        ---------
        >>> # when x & y are provided, fit_params() is called to find the best fit coefficients
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
        >>> batch = 100
        >>> x = torch.linspace(-1,1,steps=batch)
        >>> noises = torch.normal(0,1,(batch,)) * 0.02
        >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
        >>> sb.fix_symbolic(2,1,'sin',x,y)
        >>> print(sb.funs_name)
        >>> print(sb.affine[1,2,:].data)
        '''
        if isinstance(fun_name,str):
            fun = SYMBOLIC_LIB[fun_name][0]
            fun_sympy = SYMBOLIC_LIB[fun_name][1]
            fun_avoid_singularity = SYMBOLIC_LIB[fun_name][3]
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = fun_name
            
            if x == None or y == None:
                #initialzie from just fun
                self.funs[j][i] = fun
                self.funs_avoid_singularity[j][i] = fun_avoid_singularity
                if random == False:
                    self.affine.data[j][i] = torch.tensor([1.,0.,1.,0.], device=self.device)
                else:
                    self.affine.data[j][i] = torch.rand(4, device=self.device) * 2 - 1
                return None
            else:
                #initialize from x & y and fun
                params, r2 = fit_params(x,y,fun, a_range=a_range, b_range=b_range, verbose=verbose, device=self.device)
                self.funs[j][i] = fun
                self.funs_avoid_singularity[j][i] = fun_avoid_singularity
                self.affine.data[j][i] = params
                return r2
        else:
            # if fun_name itself is a function
            fun = fun_name
            fun_sympy = fun_name
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = "anonymous"

            self.funs[j][i] = fun
            self.funs_avoid_singularity[j][i] = fun
            if random == False:
                self.affine.data[j][i] = torch.tensor([1.,0.,1.,0.], device=self.device)
            else:
                self.affine.data[j][i] = torch.rand(4, device=self.device) * 2 - 1
            return None

    def get_subset(self, in_indices, out_indices):
        """
        Return a new Symbolic_KANLayer containing only the selected neurons.

        All symbolic functions (torch, sympy, names) and affine parameters
        are copied for the surviving edges so that ``symbolic_formula()``
        continues to work correctly after pruning.

        Parameters
        ----------
        in_indices : 1-D LongTensor or list of int
            Indices of the input neurons to keep.
        out_indices : 1-D LongTensor or list of int
            Indices of the output neurons to keep.

        Returns
        -------
        Symbolic_KANLayer
            A new layer with ``in_dim = len(in_indices)`` and
            ``out_dim = len(out_indices)``.
        """
        in_indices  = [int(i) for i in in_indices]
        out_indices = [int(j) for j in out_indices]

        new_sym = Symbolic_KANLayer(
            in_dim=len(in_indices), out_dim=len(out_indices), device=self.device
        )

        new_sym.mask.data   = self.mask.data[out_indices, :][:, in_indices]
        new_sym.affine.data = self.affine.data[out_indices, :, :][:, in_indices, :]

        for j_new, j_old in enumerate(out_indices):
            for i_new, i_old in enumerate(in_indices):
                new_sym.funs[j_new][i_new]                    = self.funs[j_old][i_old]
                new_sym.funs_avoid_singularity[j_new][i_new]  = self.funs_avoid_singularity[j_old][i_old]
                new_sym.funs_sympy[j_new][i_new]              = self.funs_sympy[j_old][i_old]
                new_sym.funs_name[j_new][i_new]               = self.funs_name[j_old][i_old]

        return new_sym
        
