"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import os, sys, glob
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import torchgeometry as tgm
from torch.optim import lr_scheduler
import logging
import datetime




"""
===============================================================================
basic network modules
===============================================================================
"""


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
        self.weight = self.linear.weight
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    







class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, first_omega_0=30, hidden_omega_0=30., ):
        super().__init__()
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        assert self.first_omega_0 == self.hidden_omega_0 ==30, 'they should be same in our setting'
        self.hidden_layers = hidden_layers


        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  ))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      ))
            
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    

    def forward(self, x):
        #x = x#.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        x = self.net(x)
        return x #, coords


    def jacobian(self, x):
        """analytical jacobian computation. Forward pass is first applied based on the current net params.
        This implementation passed the unit test and gives equal results with torch.autograd.jacobian upto tensor dims.
        Args:
            x (torch.Tensor): [b, in_dim]

        Raises:
            ValueError: 'to use this function, hidden layers should be 2'

        Returns:
            torch.Tensor: [b, out_dim, in_dim]
        """
        layer_inout = []
        layer_inout.append(x)
        
        with torch.no_grad():
            for layer in self.net[:-1]:
                x_curr = layer_inout[-1]
                layer_inout.append(layer(x_curr))
        derivatives = []
        for xin, layer in zip(layer_inout, self.net[:-1]):
            derivatives.append(self.first_omega_0*layer.linear.weight.unsqueeze(0)  * torch.cos(self.first_omega_0*layer.linear(xin)).unsqueeze(-1))
        grad = self.net[-1].weight.unsqueeze(0)
        for jacob in derivatives[::-1]:
            grad = grad @ jacob
        
        return grad


    


























"""
===============================================================================
basic helper functions
===============================================================================
"""

def get_logger(log_dir: str, 
               mode: str='train'):
    """create a logger to help training

    Args:
        log_dir (str): the path to save the log file
        mode (str, optional): train or test mode. Defaults to 'train'.

    Returns:
        _type_: a logger class
    """
    logger = logging.getLogger(log_dir)
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(log_dir, '{}_{}.log'.format(mode, ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.INFO)
    return logger


def get_scheduler(optimizer, policy, num_epochs_fix=None, num_epochs=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - num_epochs_fix) / float(num_epochs - num_epochs_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2)
    else:
        return NotImplementedError('scheduler with {} is not implemented'.format(policy))
    return scheduler





# ===============================================================================
# geometric transformations
# ===============================================================================


class RotConverter(nn.Module):
    """Rotation conversion between different representations

        - requires torch tensors
        
        - all functions only support data_in with [N, num_joints, D].
            - N can be n_batch, or n_batch*n_time

    """
    
    def __init__(self):
        super(RotConverter, self).__init__()

    @staticmethod
    def cont2rotmat(rotcont: torch.Tensor)->torch.Tensor:
        """Conversion from 6D representation to rotation matrix 3x3.
        
        Args:
            rotcont (torch.Tensor): [b, 6]

        Returns:
            torch.Tensor: rotation matrix, [b,3,3]
        """
        '''
        - data_in bx6
        - return: pose_matrot: bx3x3
        '''
        rotcont_ = rotcont.contiguous().view(-1, 3, 2)
        b1 = F.normalize(rotcont_[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * rotcont_[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(rotcont_[:, :, 1] - dot_prod * b1, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=2)#[b,3,3]


    @staticmethod
    def cont2rotmat_jacobian(rotcont: torch.Tensor, 
                ):
        """cont2rotmat and return analytical jacobian matrix, same to autodiff result
        This implementation runs faster than autodiff.
        
        Args:
            rotcont (torch.Tensor): [b,6]
            
        Returns:
            rotmat: torch.tensor, [b,3,3]
            jacobian matrix: torch.Tensor, [b,9,6]
        """
        rotcont_ = rotcont.contiguous().view(-1, 3, 2)
        a1 = rotcont_[:,:,0] #[b,3]
        a2 = rotcont_[:,:,1]
        nb, nd = a1.shape
        idty = torch.eye(nd).unsqueeze(0).to(a1.device)

        def ff(x):
            """batched derivative of l2 normalization

            Args:
                x (torch.Tensor): [b,d]
            """
            bb = idty- \
                torch.matmul(x.unsqueeze(-1), x.unsqueeze(1))/torch.matmul(x.unsqueeze(1), x.unsqueeze(-1))
            return bb/torch.linalg.norm(x,dim=-1,keepdim=True).unsqueeze(-1)
        
        def gg(x,y):
            """batched derivative of complement of linear projection to 
                unit vector x w.r.t. x, i.e. d((I-x*x.T)*y) / dx

            Args:
                x (torch.Tensor): [b,d]
                y (torch.Tensor): [b,d]
            """
            aa = torch.matmul(x.unsqueeze(1), y.unsqueeze(-1))*idty
            bb = torch.matmul(x.unsqueeze(-1), y.unsqueeze(1))
            return -(aa+bb)
        
        def skew(x):
            """obtain the skew-synmetric matrix of a 3D vector, batched.
            Part of the cross product representation

            Args:
                x (torch.Tensor): [b,d]
            """
            aa = torch.zeros(nb, nd, nd).to(x.device)
            aa[:,0,1] = -x[:,2]
            aa[:,0,2] = x[:,1]
            aa[:,1,0] = x[:,2]
            aa[:,1,2] = -x[:,0]
            aa[:,2,0] = -x[:,1]
            aa[:,2,1] = x[:,0]
            
            return aa
            
        
        # forward pass
        b1 = F.normalize(a1, dim=-1)
        dot_prod = torch.sum(b1 * a2, dim=1, keepdim=True)
        b2_tilde = a2 - dot_prod * b1
        b2 = F.normalize(b2_tilde, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)        
        rotmat = torch.stack([b1, b2, b3], dim=-1)#[b,3,3]
        
        ## jacobian blocks
        db1_da1 = ff(a1.detach()) #[b,3,3]
        db1_da2 = torch.zeros(nb,3,3).to(rotcont.device)
        db2_da1 = ff(b2_tilde.detach()) @ gg(b1.detach(),a2.detach()) @ db1_da1
        db2_da2 = ff(b2_tilde.detach()) @ \
            (torch.eye(nd).unsqueeze(0).to(rotcont.device)-torch.matmul(b1.unsqueeze(-1), b1.unsqueeze(1)))
        db3_da1 = skew(b2.detach()).permute(0,2,1) @ db1_da1 + skew(b1.detach())@db2_da1
        db3_da2 = skew(b2.detach()).permute(0,2,1) @ db1_da2 + skew(b1.detach())@db2_da2
        
        jacobian1 = torch.cat([db1_da1, db2_da1, db3_da1],dim=1)
        jacobian2 = torch.cat([db1_da2, db2_da2, db3_da2],dim=1)
        jacobian = torch.cat([jacobian1, jacobian2],dim=-1)
        
        # permute to fit tensor.reshape(), which cat entries row by row
        jacobian = jacobian[:,:,[0,3,1,4,2,5]] 
        jacobian = jacobian[:,[0,3,6,1,4,7,2,5,8],:] 
        
        return rotmat, jacobian # permute to fit tensor.reshape()



    @staticmethod
    def rotmat2cont(rotmat: torch.Tensor)->torch.Tensor:
        """conversion from rotation matrix to 6D.

        Args:
            rotmat (torch.Tensor): [b,3,3]

        Returns:
            torch.Tensor: [b,6]
        """
        rotmat = rotmat[:,:,:-1]
        rotmat_ = rotmat.contiguous().view(-1, 3, 2)
        return rotmat_.view(-1, 3*2)


    @staticmethod
    def aa2cont(rotaa:torch.Tensor)->torch.Tensor:
        """axis-angle to 6D rot

        Args:
            rotaa (torch.Tensor): axis angles, [b, num_joints, 3]

        Returns:
            torch.Tensor: 6D representations, [b, num_joints, 6]
        """
        
        nb = rotaa.shape[0]
        rotcont = tgm.angle_axis_to_rotation_matrix(rotaa.reshape(-1, 3))[:, :3, :2].contiguous().view(nb, -1, 6)
        
        return rotcont


    @staticmethod
    def cont2aa(rotcont: torch.Tensor)->torch.Tensor:
        """6D continuous rotation to axis-angle

        Args:
            rotcont (torch.Tensor): [b, num_joints, 6]

        Returns:
            torch.Tensor: [b, num_joints, 3]
        """
        
        batch_size = rotcont.shape[0]
        x_matrot_9d = RotConverter.cont2rotmat(rotcont).view(batch_size,-1,9)
        x_aa = RotConverter.rotmat2aa(x_matrot_9d).contiguous().view(batch_size, -1, 3)
        return x_aa


    @staticmethod
    def rotmat2aa(rotmat: torch.Tensor)->torch.Tensor:
        """from rotation matrix to axis angle

        Args:
            rotmat (torch.Tensor): [b, num_joints, 9] or [b, num_joints, 3,3]

        Returns:
            torch.Tensor: [b, num_joints, 3]
        """
        if rotmat.ndim==4:
            nt,nb = rotmat.shape[:2]
        
        homogen_rotmat = F.pad(rotmat.contiguous().view(-1, 3, 3), [0,1])
        rotaa = tgm.rotation_matrix_to_angle_axis(homogen_rotmat).contiguous().view(-1, 3)
        if rotmat.ndim==4:
            rotaa = rotaa.contiguous().view(nt, nb, 3)
        
        return rotaa


    @staticmethod
    def aa2rotmat(rotaa:torch.Tensor, flat=False)->torch.Tensor:
        """axis angle to rotation matrix

        Args:
            rotaa (torch.Tensor): [b, num_joints, 3]
            flat: (bool): flatten the rotation matrix
        Returns:
            torch.Tensor: [b, num_joints, 9]
        """
        
        nb = rotaa.shape[0]
        rotmat = tgm.angle_axis_to_rotation_matrix(rotaa.reshape(-1, 3))[:, :3, :3]
        if flat:
            rotmat = rotmat.contiguous().view(nb, -1, 9)

        return rotmat



class Transformation3D:
    """some self implemented utilities to handle transformations. Both numpy and torch are supported.
    Static type checking might be used in a future version.
    """
    @staticmethod
    def form_transf(rotmat, transl):
        """produce 4x4 transformation matrix

        Args:
            - rotmat (torch.Tensor|np.ndarray): [b,3,3]
            - transl (torch.Tensor|np.ndarray): [b,3]

        Returns:
            transf, (torch.Tensor|np.ndarray): [b,4,4]
        """
        nb = rotmat.shape[0]
        if isinstance(rotmat, torch.Tensor) and isinstance(transl, torch.Tensor):
            zeropadr = torch.zeros(nb, 1,3).to(rotmat.device)
            onepadt = torch.ones(nb, 1,1).to(rotmat.device)
            transl = transl.unsqueeze(1)
            transl = transl.permute(0,2,1) # change to (b,3,1)
            rotmat = torch.cat([rotmat, zeropadr],axis=1)
            transl = torch.cat([transl, onepadt], axis=1)
            transf = torch.cat([rotmat, transl],axis=-1)
        elif isinstance(rotmat, np.ndarray) and isinstance(transl, np.ndarray):
            zeropadr = np.zeros((nb, 1,3))
            onepadt = np.ones((nb, 1,1))
            transl = transl[:,None]
            transl = transl.transpose(0,2,1) # change to (b,3,1)
            rotmat = np.concatenate([rotmat, zeropadr],axis=1)
            transl = np.concatenate([transl, onepadt], axis=1)
            transf = np.concatenate([rotmat, transl],axis=-1)
        return transf


    @staticmethod
    def to_homo(x):
        """convert 3D points to its homogeneous coordinate

        Args:
            x (torch.Tensor|np.ndarray): [b,3]
            
        Returns:
            x_homo: [b,4]
        """
        if isinstance(x, torch.Tensor):
            return F.pad(x, (0,1), 'constant', 1)
        elif isinstance(x, np.ndarray):
            return np.pad(x, ((0,0),(0,1)), 'constant', constant_values=1)
        
        
    @staticmethod
    def transform(transf, x):
        """perform transformation

        Args:
            transf (torch.Tensor)|np.ndarray: [b,4,4]
            x (torch.Tensor|np.ndarray): [b,3]

        Returns:
            xt, torch.Tensor, [b,3]
        """
        x_homo = Transformation3D.to_homo(x)
        if isinstance(x_homo, torch.Tensor):
            xt_homo = torch.einsum('bij,bj->bi',transf, x_homo)
        elif isinstance(x_homo, np.ndarray):
            xt_homo = np.einsum('bij,bj->bi',transf, x_homo)
        return xt_homo[:,:-1] # from homo to 3D
        





