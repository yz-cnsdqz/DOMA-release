import math
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev, vmap, jacfwd

from pytorch3d.loss import chamfer_distance
from pytorch3d import ops
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import IO



from models.baseops import Siren, Transformation3D
from models.baseops import get_scheduler
from models.baseops import RotConverter

from utils.vis import *
from utils.metrics import *








"""
===============================================================================
network modules
===============================================================================
"""


class DeformSiren(torch.nn.Module):
    def __init__(self, opt):
        super(DeformSiren, self).__init__()
        self.siren = Siren(in_features=opt.in_features,
                           hidden_features=opt.hidden_features,
                           hidden_layers=opt.n_hidden_layers,
                           out_features=3,
                           outermost_linear=True,)

        self.opt = opt

    def forward(self, x):
        x = self.siren(x) + x
        return x


    def jacobian(self, x):
        jacob =self.siren.jacobian(x)
        return jacob


class DeformSiren4D(torch.nn.Module):
    def __init__(self, opt):
        super(DeformSiren4D, self).__init__()
        self.siren = Siren(in_features=opt.in_features,
                           hidden_features=opt.hidden_features,
                           hidden_layers=opt.n_hidden_layers,
                           out_features=opt.out_features,
                           outermost_linear=True,)
        ts = torch.linspace(-1,1,opt.max_lens)
        self.register_buffer("ts", ts)
        self.opt = opt

    def forward(self, x,tidx):
        if tidx.dim()==0:
            ts = self.ts[tidx].unsqueeze(0).repeat(x.shape[0], 1)
        else:
            ts = tidx

        xs = torch.cat([x, ts],dim=-1)

        return self.siren(xs) + x[...,:3]


    def jacobian(self, x,tidx):
        if tidx.dim()==0:
            ts = self.ts[tidx].unsqueeze(0).repeat(x.shape[0], 1)
        else:
            ts = tidx
        xs = torch.cat([x, ts],dim=-1)
        jacob =self.siren.jacobian(xs)
        return jacob[...,:-1]


class DeformSirenSE3(torch.nn.Module):
    """ similar to nerfies"""
    def __init__(self, opt):
        super().__init__()
        self.siren = Siren(in_features=opt.in_features,
                           hidden_features=opt.hidden_features,
                           hidden_layers=opt.n_hidden_layers,
                           out_features=opt.out_features,
                           outermost_linear=True,)
        ts = torch.linspace(-1,1,opt.max_lens)
        self.register_buffer("ts", ts)
        self.opt = opt


    def forward(self, x,ts):
        xs = torch.cat([x, ts],dim=-1)
        transf = self.siren(xs)
        rotcont, transl = transf[...,:6], transf[...,6:]
        # change axis-angle to rotation matrix
        rotmat = RotConverter.cont2rotmat(rotcont)

        return rotmat, transl

    def jacobian(self, x, tidx):
        ts = self.ts[tidx].unsqueeze(0).repeat(x.shape[0], 1)
        xs = torch.cat([x, ts],dim=-1)
        jacob = self.siren.jacobian(xs)
        return jacob[...,:-1] # return spatial derivatives


class DeformSirenScaledSE3(torch.nn.Module):
    """ similar to nerfies"""
    def __init__(self, opt):
        super().__init__()
        self.siren = Siren(in_features=opt.in_features,
                           hidden_features=opt.hidden_features,
                           hidden_layers=opt.n_hidden_layers,
                           out_features=opt.out_features,
                           outermost_linear=True,)
        ts = torch.linspace(-1,1,opt.max_lens)
        self.register_buffer("ts", ts)
        self.opt = opt


    def forward(self, x,ts):
        # ts = self.ts[tidx].unsqueeze(0).repeat(x.shape[0], 1)
        xs = torch.cat([x, ts],dim=-1)
        transf = self.siren(xs)
        scale, rotcont, transl = transf[:,:1], transf[...,1:7], transf[...,7:]
        # change axis-angle to rotation matrix
        rotmat = RotConverter.cont2rotmat(rotcont)
        scale = F.softplus(scale)

        return scale, rotmat, transl

    def jacobian(self, x, tidx):
        ts = self.ts[tidx].unsqueeze(0).repeat(x.shape[0], 1)
        xs = torch.cat([x, ts],dim=-1)
        jacob = self.siren.jacobian(xs)
        return jacob[...,:-1] # return spatial derivatives



class DeformSirenAffine(torch.nn.Module):
    """ similar to nerfies"""
    def __init__(self, opt):
        super().__init__()
        self.siren = Siren(in_features=opt.in_features,
                           hidden_features=opt.hidden_features,
                           hidden_layers=opt.n_hidden_layers,
                           out_features=opt.out_features,
                           outermost_linear=True,)
        ts = torch.linspace(-1,1,opt.max_lens)
        self.register_buffer("ts", ts)
        self.opt = opt
        assert opt.out_features == 12, "wrong output dimension"

    def forward(self, x,ts):
        # ts = self.ts[tidx].unsqueeze(0).repeat(x.shape[0], 1)
        xs = torch.cat([x, ts],dim=-1)
        transf = self.siren(xs)
        affine, transl = transf[...,:9], transf[...,9:]

        affine = affine.reshape(-1,3,3)

        return affine, transl


    def jacobian(self, x, tidx):
        ts = self.ts[tidx].unsqueeze(0).repeat(x.shape[0], 1)
        xs = torch.cat([x, ts],dim=-1)
        jacob = self.siren.jacobian(xs)
        return jacob[...,:-1] # return spatial derivatives





"""
===============================================================================
Motion Field Models
===============================================================================
"""



class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])



class BoneCloud(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        # initialize bones
        self.n_bones = n_bones = opt['n_bones']
        self.nt = nt = opt['n_frames']
        self.sigma = opt['sigma']

        if opt['bone_sample'] == 'rand':
            xyz_min, xyz_max = opt['xyz_min'], opt['xyz_max']
            bone_locs = torch.rand(n_bones,3)*(xyz_max-xyz_min) + xyz_min
        elif opt['bone_sample'] == 'grid':
            print('#bones are set to 10 per dim')
            xyz_min, xyz_max = opt['xyz_min'], opt['xyz_max']
            xx = torch.linspace(-xyz_min[0], xyz_max[0], 10)
            yy = torch.linspace(-xyz_min[1], xyz_max[1], 10)
            zz = torch.linspace(-xyz_min[2], xyz_max[2], 10)
            X, Y, Z = torch.meshgrid(xx, yy, zz, indexing='ij')
            bone_locs = torch.stack([X,Y,Z],dim=-1).reshape(-1,3)

        transl_init = torch.zeros(nt, n_bones, 3)
        rotmat_init = torch.diag_embed(torch.ones(nt, n_bones, 3))
        rotcont_init = rotmat_init[...,:-1].reshape(nt,-1,6)
        bone_transf = torch.cat([rotcont_init, transl_init],dim=-1).detach()

        if opt['optimize_bone_locs']:
            self.bone_locs = torch.nn.Parameter(bone_locs)
        else:
            self.register_buffer('bone_locs', bone_locs)

        self.bone_transf=torch.nn.Parameter(bone_transf)


    def lbs(self,xyz_c, tidx, transform_normals=False):
        # compute skinning weights
        skw = torch.exp(-self.sigma*torch.cdist(xyz_c, self.bone_locs))
        skw = skw / skw.sum(dim=-1,keepdim=True)
        # compute transform
        if transform_normals:
            rotcont= self.bone_transf[tidx,...,:6]
            rotmat = RotConverter.cont2rotmat(rotcont)
            xyz_pred = torch.einsum('bk,kij,bj->bi',skw, rotmat,xyz_c)
            xyz_pred = F.normalize(xyz_pred, 2, dim=-1)
        else:
            rotcont, transl = self.bone_transf[tidx,...,:6],self.bone_transf[tidx,...,6:]
            rotmat = RotConverter.cont2rotmat(rotcont)
            transf = Transformation3D.form_transf(rotmat, transl)
            xyz_c_homo = Transformation3D.to_homo(xyz_c)
            xyz_pred = torch.einsum('bk,kij,bj->bi',skw, transf,xyz_c_homo)[...,:-1]
        return xyz_pred


    def train_field(self, body_verts, body_verts_mask):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt['lr'])
        n_iter = self.opt['n_iter']
        scheduler = get_scheduler(optimizer, policy='lambda',
                                num_epochs_fix=10,
                                num_epochs=n_iter)

        xyz_c = body_verts[0]

        ## optimizing bone cloud
        for ii in range(n_iter):
            tss = torch.randperm(self.nt)
            loss_info = 0
            for tidx in tss:
                xyz_t = body_verts[tidx].detach()
                mask = body_verts_mask[tidx]
                xyz_pred = self.lbs(xyz_c, tidx)
                loss_rec = F.l1_loss(xyz_pred*mask, xyz_t*mask)

                loss = loss_rec

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_info += loss_rec.item() / tss.shape[0]

            if ii % 100 == 0 or ii==n_iter-1:
                print(f"iter={ii}, loss={loss_info:03f}")

            scheduler.step()







class SE3Field4D(nn.Module):
    """this takes the insight of nerfies: 1) SE3 field, 2) elasticity regularization
    Rather than MLP, we use siren networks as in dpf, and this is the 4D version.
    """
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.nt = nt = opt['n_frames']
        opt['dpf_opt']['out_features'] = 9
        # define network
        self.se3net = DeformSirenSE3(Dict2Class(opt['dpf_opt']))

    def german_mcclure(self, x, c=0.03):
        div2 = (x / c).pow(2)
        return 2*div2 / (div2 + 4)


    def deform(self, xyz_c, tidx):
        ts = self.se3net.ts[tidx].unsqueeze(0).repeat(xyz_c.shape[0], 1)
        xyz_pred = self.deform_(xyz_c, ts)
        return xyz_pred


    def deform_(self, xyz_c, ts):
        rotmat, transl = self.se3net(xyz_c, ts)
        xyz_pred = torch.bmm(rotmat, xyz_c.unsqueeze(2))[...,0] + transl
        return xyz_pred

    def elasticity_reg(self,xyz, tidx):
        ts = self.se3net.ts[tidx].unsqueeze(0).repeat(xyz.shape[0], 1)
        jacob = vmap(jacrev(self.deform_, argnums=0))(xyz.unsqueeze(1),ts.unsqueeze(1))
        jacob = jacob.squeeze()
        U, S, Vh = torch.linalg.svd(jacob, full_matrices=False)
        loss_elastic = self.german_mcclure(S.log().pow(2).sum(-1)).mean()

        return loss_elastic



    def homogenous_reg(self, xyz, tidx, norm='l2'):
        jacob = self.se3net.jacobian(xyz, tidx)
        if norm == 'l2':
            loss_homo = jacob.pow(2).sum(dim=(-2,-1)).mean()
        elif norm == 'charbonnier':
            loss_homo = ((jacob.pow(2) + 1.).sqrt()-1.).sum(dim=(-2,-1)).mean()
        return loss_homo


    def train_field(self, body_verts, body_verts_mask):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt['lr'])
        n_iter = self.opt['n_iter']
        scheduler = get_scheduler(optimizer, policy='lambda',
                                num_epochs_fix=10,
                                num_epochs=n_iter)

        mask_c = body_verts_mask[0]
        xyz_c = body_verts[0]


        ## optimizing bone cloud
        for ii in range(n_iter):
            tss = torch.randperm(self.nt)
            loss_info = 0
            for tidx in tss:
                xyz_t = body_verts[tidx].detach()
                mask = body_verts_mask[tidx] * mask_c
                xyz_pred = self.deform(xyz_c, tidx)
                loss_rec =  F.l1_loss(mask*xyz_pred, mask*xyz_t)
                if self.opt['elastic_loss_weight']>0:
                    loss_elasticity = self.elasticity_reg(xyz_c, tidx)
                else:
                    loss_elasticity = 0

                if self.opt['homo_loss_weight']>0:
                    loss_homo = self.homogenous_reg(xyz_c, tidx)
                else:
                    loss_homo = 0

                loss = loss_rec + self.opt['elastic_loss_weight']*loss_elasticity + self.opt['homo_loss_weight']*loss_homo


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_info += loss_rec.item() / tss.shape[0]

            if ii % 100 == 0 or ii==n_iter-1:
                print(f"trainfield, iter={ii}, loss={loss_info:03f}")

            scheduler.step()



class ScaledSE3Field4D(nn.Module):
    """this takes the insight of nerfies: 1) SE3 field, 2) elasticity regularization
    Rather than MLP, we use siren networks as in dpf, and this is the 4D version.
    """
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.nt = nt = opt['n_frames']
        opt['dpf_opt']['out_features'] = 10
        # define network
        self.se3net = DeformSirenScaledSE3(Dict2Class(opt['dpf_opt']))

    def german_mcclure(self, x, c=0.03):
        div2 = (x / c).pow(2)
        return 2*div2 / (div2 + 4)


    def deform(self, xyz_c, tidx):
        ts = self.se3net.ts[tidx].unsqueeze(0).repeat(xyz_c.shape[0], 1)
        xyz_pred = self.deform_(xyz_c, ts)
        return xyz_pred


    def deform_(self, xyz_c, ts):
        scale, rotmat, transl = self.se3net(xyz_c, ts)
        xyz_pred = scale * torch.bmm(rotmat, xyz_c.unsqueeze(2))[...,0] + transl
        return xyz_pred

    def elasticity_reg(self,xyz, tidx):
        ts = self.se3net.ts[tidx].unsqueeze(0).repeat(xyz.shape[0], 1)
        jacob = vmap(jacrev(self.deform_, argnums=0))(xyz.unsqueeze(1),ts.unsqueeze(1))
        jacob = jacob.squeeze()
        U, S, Vh = torch.linalg.svd(jacob, full_matrices=False)
        loss_elastic = self.german_mcclure(S.log().pow(2).sum(-1)).mean()

        return loss_elastic



    def homogenous_reg(self, xyz, tidx, norm='l2'):
        jacob = self.se3net.jacobian(xyz, tidx)
        if norm == 'l2':
            loss_homo = jacob.pow(2).sum(dim=(-2,-1)).mean()
        elif norm == 'charbonnier':
            loss_homo = ((jacob.pow(2) + 1.).sqrt()-1.).sum(dim=(-2,-1)).mean()
        return loss_homo


    def train_field(self, body_verts, body_verts_mask):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt['lr'])
        n_iter = self.opt['n_iter']
        scheduler = get_scheduler(optimizer, policy='lambda',
                                num_epochs_fix=10,
                                num_epochs=n_iter)

        mask_c = body_verts_mask[0]
        xyz_c = body_verts[0]


        ## optimizing bone cloud
        for ii in range(n_iter):
            tss = torch.randperm(self.nt)
            loss_info = 0
            for tidx in tss:
                xyz_t = body_verts[tidx].detach()
                mask = body_verts_mask[tidx] * mask_c
                xyz_pred = self.deform(xyz_c, tidx)
                loss_rec =  F.l1_loss(mask*xyz_pred, mask*xyz_t)
                if self.opt['elastic_loss_weight']>0:
                    loss_elasticity = self.elasticity_reg(xyz_c, tidx)
                else:
                    loss_elasticity = 0

                if self.opt['homo_loss_weight']>0:
                    loss_homo = self.homogenous_reg(xyz_c, tidx)
                else:
                    loss_homo = 0

                loss = loss_rec + self.opt['elastic_loss_weight']*loss_elasticity + self.opt['homo_loss_weight']*loss_homo

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_info += loss_rec.item() / tss.shape[0]

            if ii % 100 == 0 or ii==n_iter-1:
                print(f"trainfield, iter={ii}, loss={loss_info:03f}")

            scheduler.step()



class AffineField4D(nn.Module):
    """this takes the insight of nerfies: 1) affine field, 2) elasticity regularization
    Rather than MLP, we use siren networks as in dpf, and this is the 4D version.
    """
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.nt = nt = opt['n_frames']
        opt['dpf_opt']['out_features'] = 12
        # define network
        self.affinenet = DeformSirenAffine(Dict2Class(opt['dpf_opt']))

    def german_mcclure(self, x, c=0.03):
        div2 = (x / c).pow(2)
        return 2*div2 / (div2 + 4)


    def deform(self, xyz_c, tidx):
        ts = self.affinenet.ts[tidx].unsqueeze(0).repeat(xyz_c.shape[0], 1)
        xyz_pred = self.deform_(xyz_c, ts)
        return xyz_pred

    def deform_(self, xyz_c, ts):
        affine, transl = self.affinenet(xyz_c, ts)
        xyz_pred = torch.bmm(affine, xyz_c.unsqueeze(2))[...,0] + transl
        return xyz_pred


    def elasticity_reg(self,xyz, tidx):
        ts = self.affinenet.ts[tidx].unsqueeze(0).repeat(xyz.shape[0], 1)
        jacob = vmap(jacrev(self.deform_, argnums=0))(xyz.unsqueeze(1),ts.unsqueeze(1))
        jacob = jacob.squeeze()
        U, S, Vh = torch.linalg.svd(jacob, full_matrices=False)
        loss_elastic = self.german_mcclure(S.log().pow(2).sum(-1)).mean()
        return loss_elastic


    def homogenous_reg(self, xyz, tidx, norm='l2', use_autodiff=False):
        if use_autodiff:
            ts = self.affinenet.ts[tidx].unsqueeze(0).repeat(xyz.shape[0], 1)
            jacob_A, jacob_u = vmap(jacrev(self.affinenet, argnums=0))(xyz.unsqueeze(1),ts.unsqueeze(1))
            jacob_A = jacob_A.squeeze().reshape(-1,9,3)
            jacob_u = jacob_u.squeeze()
            jacob = torch.cat([jacob_A, jacob_u],dim=1)
        else:
            jacob = self.affinenet.jacobian(xyz, tidx)

        if norm == 'l2':
            loss_homo = jacob.pow(2).sum(dim=(-2,-1)).mean()
        elif norm == 'charbonnier':
            loss_homo = ((jacob.pow(2) + 1.).sqrt()-1.).sum(dim=(-2,-1)).mean()
        return loss_homo


    def train_field(self, body_verts, body_verts_mask):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt['lr'])
        n_iter = self.opt['n_iter']
        scheduler = get_scheduler(optimizer, policy='lambda',
                                num_epochs_fix=10,
                                num_epochs=n_iter)

        mask_c = body_verts_mask[0]
        xyz_c = body_verts[0]


        ## optimizing bone cloud
        for ii in range(n_iter):
            tss = torch.randperm(self.nt)
            loss_info = 0
            for tidx in tss:
                xyz_t = body_verts[tidx].detach()
                mask = body_verts_mask[tidx] * mask_c
                xyz_pred = self.deform(xyz_c, tidx)
                loss_rec =  F.l1_loss(mask*xyz_pred, mask*xyz_t)
                if self.opt['elastic_loss_weight']>0:
                    loss_elasticity = self.elasticity_reg(xyz_c, tidx)
                else:
                    loss_elasticity = 0

                if self.opt['homo_loss_weight']>0:
                    iidx = torch.randperm(xyz_c.shape[0])[:1024]
                    xyz_c_ = xyz_c[iidx] + 0.1*torch.randn_like(xyz_c[iidx])
                    loss_homo = self.homogenous_reg(xyz_c_, tidx, norm='l2')
                else:
                    loss_homo = 0

                loss = loss_rec + self.opt['elastic_loss_weight']*loss_elasticity + self.opt['homo_loss_weight']*loss_homo


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_info += loss_rec.item() / tss.shape[0]

            if ii % 100 == 0 or ii==n_iter-1:
                print(f"trainfield, iter={ii}, loss={loss_info:03f}")

            scheduler.step()



class Transfield(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        opt['dpf_opt']['out_features'] = 3
        self.dpf = DeformSiren4D(Dict2Class(opt['dpf_opt']))
        self.nt = opt['n_frames']


    def german_mcclure(self, x, c=0.03):
        div2 = (x / c).pow(2)
        return 2*div2 / (div2 + 4)

    def elasticity_reg(self,xyz, tidx):
        ts = self.dpf.ts[tidx].unsqueeze(0).repeat(xyz.shape[0], 1)
        jacob = vmap(jacrev(self.dpf, argnums=0))(xyz.unsqueeze(1),ts.unsqueeze(1))
        jacob = jacob.squeeze()
        U, S, Vh = torch.linalg.svd(jacob, full_matrices=False)
        loss_elastic = self.german_mcclure(S.log().pow(2).sum(-1)).mean()

        return loss_elastic


    def homogenous_reg(self, xyz, tidx, norm='l2'):
        jacob = self.dpf.jacobian(xyz, tidx) # only spatial jacobian
        if norm == 'l2':
            loss_homo = jacob.pow(2).sum(dim=(-2,-1)).mean()
        elif norm == 'charbonnier':
            loss_homo = ((jacob.pow(2) + 1.).sqrt()-1.).sum(dim=(-2,-1)).mean()

        return loss_homo



    def train_field(self, body_verts, body_verts_mask):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt['lr'])
        n_iter = self.opt['n_iter']
        scheduler = get_scheduler(optimizer, policy='lambda',
                                num_epochs_fix=10,
                                num_epochs=n_iter)

        mask_c = body_verts_mask[0]
        xyz_c = body_verts[0]


        ## optimizing bone cloud
        for ii in range(n_iter):
            tss = torch.randperm(self.nt)
            loss_info = 0
            for tidx in tss:
                xyz_t = body_verts[tidx].detach()
                mask = body_verts_mask[tidx] * mask_c
                xyz_pred = self.dpf(xyz_c, tidx)
                loss_rec =  F.l1_loss(mask*xyz_pred, mask*xyz_t)
                if self.opt['elastic_loss_weight']>0:
                    loss_elasticity = self.elasticity_reg(xyz_c, tidx)
                else:
                    loss_elasticity = 0

                if self.opt['homo_loss_weight']>0:
                    loss_homo = self.homogenous_reg(xyz_c, tidx)
                else:
                    loss_homo = 0

                loss = loss_rec + self.opt['elastic_loss_weight']*loss_elasticity + self.opt['homo_loss_weight']*loss_homo

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_info += loss_rec.item() / tss.shape[0]

            if ii % 100 == 0 or ii==n_iter-1:
                print(f"trainfield, iter={ii}, loss={loss_info:03f}")

            scheduler.step()



class DPFBAG(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.nt = nt = opt['n_frames']
        opt['dpf_opt']['in_features'] = 3
        opt['dpf_opt']['out_features'] = 3
        self.dpf = nn.ModuleList(
            [DeformSiren(Dict2Class(opt['dpf_opt'])) for _ in range(nt)])


    def german_mcclure(self, x, c=0.03):
        div2 = (x / c).pow(2)
        return 2*div2 / (div2 + 4)


    def elasticity_reg(self,xyz, tidx):
        jacob = vmap(jacrev(self.dpf[tidx], argnums=0))(xyz.unsqueeze(1))
        jacob = jacob.squeeze()
        U, S, Vh = torch.linalg.svd(jacob, full_matrices=False)
        loss_elastic = self.german_mcclure(S.log().pow(2).sum(-1)).mean()

        return loss_elastic


    def homogenous_reg(self, xyz, tidx, norm='l2'):
        jacob = self.dpf[tidx].jacobian(xyz) # only spatial jacobian
        if norm == 'l2':
            loss_homo = jacob.pow(2).sum(dim=(-2,-1)).mean()
        elif norm == 'charbonnier':
            loss_homo = ((jacob.pow(2) + 1.).sqrt()-1.).sum(dim=(-2,-1)).mean()
        return loss_homo


    def deform(self, xyz, tidx):
        xyz_pred = self.dpf[tidx](xyz)
        return xyz_pred


    def train_field(self, body_verts, body_verts_mask):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt['lr'])
        n_iter = self.opt['n_iter']
        scheduler = get_scheduler(optimizer, policy='lambda',
                                num_epochs_fix=10,
                                num_epochs=n_iter)

        mask_c = body_verts_mask[0]
        xyz_c = body_verts[0]


        ## optimizing bone cloud
        for ii in range(n_iter):
            tss = torch.randperm(self.nt)
            loss_info = 0
            for tidx in tss:
                xyz_t = body_verts[tidx].detach()
                mask = body_verts_mask[tidx] * mask_c
                xyz_pred = self.deform(xyz_c, tidx)
                loss_rec =  F.l1_loss(mask*xyz_pred, mask*xyz_t)
                if self.opt['elastic_loss_weight']>0:
                    loss_elasticity = self.elasticity_reg(xyz_c, tidx)
                else:
                    loss_elasticity = 0

                if self.opt['homo_loss_weight']>0:
                    loss_homo = self.homogenous_reg(xyz_c, tidx)
                else:
                    loss_homo = 0

                loss = loss_rec + self.opt['elastic_loss_weight']*loss_elasticity + self.opt['homo_loss_weight']*loss_homo

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_info += loss_rec.item() / tss.shape[0]

            if ii % 100 == 0 or ii==n_iter-1:
                print(f"trainfield, iter={ii}, loss={loss_info:03f}")

            scheduler.step()



from third_parties.banmo.nerf import NeRF, RTHead, Embedding
from third_parties.banmo.geom_utils import (generate_bones, warp_bw, bone_transform,
                                            rtk_invert, rtk_compose,
                                            gauss_mlp_skinning, lbs)
class BANMO(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        num_freqs = opt['n_freqs'] # as in banmo code
        t_embed_dim = opt['t_embed_dim'] # default in banmo
        self.num_bones = num_bones = opt['n_bones'] # default in banmo
        self.nt = n_frames = opt['n_frames']

        # some used embeddings in banmo
        self.embedding_xyz = Embedding(3,num_freqs,alpha=None)
        self.framecode_fourier_embed = Embedding(1,num_freqs,alpha=num_freqs)
        self.framecode_basis_mlp = nn.Linear(self.framecode_fourier_embed.out_channels,
                                t_embed_dim)
        self.pose_code = nn.Sequential(self.framecode_fourier_embed,
                                       self.framecode_basis_mlp)
        self.rest_pose_code = nn.Embedding(1, t_embed_dim)

        ## about poses
        self.nerf_body_rts = nn.Sequential(self.pose_code,
                            RTHead(use_quat=False,
                            #D=5,W=128,
                            in_channels_xyz=t_embed_dim,in_channels_dir=0,
                            out_channels=6*num_bones, raw_feat=True))


        ## about bone
        bones= generate_bones(num_bones, num_bones, 0 )
        self.bones = nn.Parameter(bones)

        # about skinning
        in_channels_xyz=3+3*num_freqs*2
        self.nerf_skin = NeRF(in_channels_xyz=in_channels_xyz+t_embed_dim,
                                    D=5,W=64,
                     in_channels_dir=0, out_channels=num_bones,
                     raw_feat=True, in_channels_code=t_embed_dim)
        skin_aux = torch.Tensor([0,1])
        self.skin_aux = nn.Parameter(skin_aux)

    def correct_bones(self, bones_rst, ):
        # bones=>bones_rst
        bones_rst = bones_rst.clone()
        rest_pose_code =  self.rest_pose_code
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))
        rts_head = self.nerf_body_rts[1]
        bone_rts_rst = rts_head(rest_pose_code)[0] # 1,B*12
        bones_rst = bone_transform(bones_rst, bone_rts_rst, is_vec=True)[0]
        return bones_rst, bone_rts_rst

    def correct_rest_pose(self, bone_rts_fw, bone_rts_rst):
        # delta rts
        # bone_rts_fw = bone_rts_fw.clone()
        rts_shape = bone_rts_fw.shape
        bone_rts_rst_inv = rtk_invert(bone_rts_rst, self.num_bones)
        bone_rts_rst_inv = bone_rts_rst_inv.repeat(rts_shape[0],rts_shape[1],1)
        bone_rts_fw =     rtk_compose(bone_rts_rst_inv, bone_rts_fw)
        return bone_rts_fw



    def deform(self, xyz_c, tidx):
        """restructure warp_fw in banmo.geom_utils
        """
        nb = xyz_c.shape[0]
        query_time = torch.ones(1,1).long().to(xyz_c.device)*tidx
        bones_rst = self.bones
        bone_rts_fw = self.nerf_body_rts(query_time)
        bones_rst, bone_rts_rst = self.correct_bones(bones_rst)
        bone_rts_fw = self.correct_rest_pose(bone_rts_fw, bone_rts_rst)

        rest_pose_code =  self.rest_pose_code
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))

        skin_forward = gauss_mlp_skinning(xyz_c.unsqueeze(0), self.embedding_xyz, bones_rst.unsqueeze(0), rest_pose_code.unsqueeze(0), self.nerf_skin, skin_aux=self.skin_aux)

        pts_dfm, _ = lbs(bones_rst, bone_rts_fw, skin_forward, xyz_c,backward=False)
        # pts_dfm = pts_dfm[:,0]
        # vertices = pts_dfm.cpu().numpy()
        return pts_dfm[0]



    def train_field(self, body_verts, body_verts_mask):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt['lr'])
        n_iter = self.opt['n_iter']
        scheduler = get_scheduler(optimizer, policy='lambda',
                                num_epochs_fix=10,
                                num_epochs=n_iter)

        mask_c = body_verts_mask[0]
        xyz_c = body_verts[0]


        ## optimizing bone cloud
        for ii in range(n_iter):
            tss = torch.randperm(self.nt)
            loss_info = 0
            for tidx in tss:
                xyz_t = body_verts[tidx].detach()
                mask = body_verts_mask[tidx] * mask_c
                xyz_pred = self.deform(xyz_c, tidx)
                loss_rec =  F.l1_loss(mask*xyz_pred, mask*xyz_t)

                loss = loss_rec

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_info += loss_rec.item() / tss.shape[0]

            if ii % 100 == 0 or ii==n_iter-1:
                print(f"iter={ii}, loss={loss_info:03f}")

            scheduler.step()






"""
===============================================================================
Application: Novel Point Motion Prediction
===============================================================================
"""

class MotionField(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        motion_model_opt = opt['motion_model_opt']
        motion_model_name = opt['motion_model_name']
        motion_model_opt['device'] = opt['device']
        if motion_model_name == 'bonecloud':
            self.model = BoneCloud(motion_model_opt)
            self.deform = self.model.lbs
        elif motion_model_name == 'se3field4d':
            self.model = SE3Field4D(motion_model_opt)
            self.deform = self.model.deform
        elif motion_model_name == 'scaledse3field4d':
            self.model = ScaledSE3Field4D(motion_model_opt)
            self.deform = self.model.deform
        elif motion_model_name == 'affinefield4d':
            self.model = AffineField4D(motion_model_opt)
            self.deform = self.model.deform
        elif motion_model_name == 'transfield4d':
            self.model = Transfield(motion_model_opt)
            self.deform = self.model.dpf
        elif motion_model_name == 'dpfbag':
            self.model = DPFBAG(motion_model_opt)
            self.deform = self.model.deform
        elif motion_model_name == 'banmo':
            self.model = BANMO(motion_model_opt)
            self.deform = self.model.deform


    def train_motion_field(self, train_pts, outputfolder):
        train_pts_mask = torch.ones_like(train_pts)
        self.model.train_field(train_pts, train_pts_mask)
        torch.save({'model_state_dict': self.model.state_dict()}, outputfolder+'/last.pt')


    def test_motion_field(self, test_pts, test_normals, outputfolder, 
                          vis=True):
        self.model.load_state_dict(torch.load(outputfolder+'/last.pt')['model_state_dict'])
        

        pcd_c = test_pts[0]
        pcd_pred = []
        with torch.no_grad():
            tss = torch.arange(self.opt['motion_model_opt']['n_frames'])
            for tidx in tss:
                xyz_pred = self.deform(pcd_c, tidx)
                pcd_pred.append(xyz_pred)
            pcd_pred = torch.stack(pcd_pred)
            gt_flow = test_pts-test_pts[:1]
            pred_flow = pcd_pred-test_pts[:1]
            l1flowdist = torch.abs(pred_flow - gt_flow).norm(1, dim=-1).mean().item()
            l1dist = torch.abs(pcd_pred - test_pts).norm(1, dim=-1).mean().item()


        results = self.opt
        results['test_flow_l1norm_mean'] = l1flowdist
        results['test_loc_l1norm_mean'] = l1dist

        with open(f"{outputfolder}/test.yaml", 'w') as file:
            yaml.dump(results, file)


        if vis:
            vis_seq_pytorch3d(pcd_pred, normals_to_rgb(test_normals),f'{outputfolder}/test')
            vis_seq_pytorch3d(test_pts, normals_to_rgb(test_normals),f'{outputfolder}/gt')



    def eval_mesh_deformation(self, meshes, outputfolder):
        self.model.load_state_dict(torch.load(outputfolder+'/last.pt')['model_state_dict'],
                                   )
        pcd_c = meshes.verts_list()[0]
        faces = meshes.faces_list()
        pcd_pred = []
        with torch.no_grad():
            tss = torch.arange(self.opt['motion_model_opt']['n_frames'])
            for tidx in tss:
                xyz_pred = self.deform(pcd_c, tidx)
                pcd_pred.append(xyz_pred)
            
            meshes_pred = Meshes(verts=pcd_pred, faces=faces).to(pcd_c.device)
            
        l2_chamfer, l2_normals = eval_surface_reconstruction_mesh(meshes_pred, meshes)

        results = self.opt
        results['l2_chamfer'] = l2_chamfer.item()
        results['l2_normals'] = l2_normals.item()

        with open(f"{outputfolder}/test_meshdeform.yaml", 'w') as file:
            yaml.dump(results, file)









"""
===============================================================================
Application: Guided Mesh Alignment
===============================================================================
"""

class RegistrationOP(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        motion_model_opt = opt['motion_model_opt']
        self.motion_model_name = motion_model_name = opt['motion_model_name']

        motion_model_opt['device'] = opt['device']
        if motion_model_name == 'bonecloud':
            self.model = BoneCloud(motion_model_opt)
            self.deform = self.model.lbs
        elif motion_model_name == 'se3field4d':
            self.model = SE3Field4D(motion_model_opt)
            self.deform = self.model.deform
        elif motion_model_name == 'scaledse3field4d':
            self.model = ScaledSE3Field4D(motion_model_opt)
            self.deform = self.model.deform
        elif motion_model_name == 'affinefield4d':
            self.model = AffineField4D(motion_model_opt)
            self.deform = self.model.deform
        elif motion_model_name == 'transfield4d':
            self.model = Transfield(motion_model_opt)
            self.deform = self.model.dpf
        elif motion_model_name == 'dpfbag':
            self.model = DPFBAG(motion_model_opt)
            self.deform = self.model.deform
        elif motion_model_name == 'banmo':
            self.model = BANMO(motion_model_opt)
            self.deform = self.model.deform




    def aiap_loss(self, x_canonical, x_deformed, n_neighbors=5):
        """
        Computes the as-isometric-as-possible loss between two sets of points, which measures the discrepancy
        between their pairwise distances.

        Parameters
        ----------
        x_canonical : array-like, shape (n_points, n_dims)
            The canonical (reference) point set, where `n_points` is the number of points
            and `n_dims` is the number of dimensions.
        x_deformed : array-like, shape (n_points, n_dims)
            The deformed (transformed) point set, which should have the same shape as `x_canonical`.
        n_neighbors : int, optional
            The number of nearest neighbors to use for computing pairwise distances.
            Default is 5.

        Returns
        -------
        loss : float
            The AIAP loss between `x_canonical` and `x_deformed`, computed as the L1 norm
            of the difference between their pairwise distances. The loss is a scalar value.
        Raises
        ------
        ValueError
            If `x_canonical` and `x_deformed` have different shapes.
        """

        if x_canonical.shape != x_deformed.shape:
            raise ValueError("Input point sets must have the same shape.")

        _, nn_ix, _ = ops.knn_points(x_canonical.unsqueeze(0),
                                x_canonical.unsqueeze(0),
                                K=n_neighbors,
                                return_sorted=True)

        dists_canonical = torch.cdist(x_canonical[nn_ix], x_canonical[nn_ix])
        dists_deformed = torch.cdist(x_deformed[nn_ix], x_deformed[nn_ix])

        loss = F.l1_loss(dists_canonical, dists_deformed)

        return loss



    def registration(self, body_verts,  body_pcds, outputfolder):
        """register the body_scans[0] to body_pcds, guided by body_verts
        """
        with open(f"{outputfolder}/opt.yaml", 'w') as file:
            yaml.dump(self.opt, file)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt['lr'])
        n_iter = self.opt['n_iter']
        scheduler = get_scheduler(optimizer, policy='lambda',
                                num_epochs_fix=0.1*self.opt['lr'],
                                num_epochs=n_iter)



        ## optimizing bone cloud sequentially, otherwise chamfer dist will not work well
        verts_c = body_verts[0].detach()
        for ii in range(n_iter):
            tss = torch.randperm(self.model.nt)
            info_losses = {'loss_guide':0, 'loss_cham':0, 'loss_rigid':0,
                           'loss_homo':0, 'loss_aiap':0}
            for tidx in tss:
                verts_t = body_verts[tidx].detach()
                iidx = torch.randperm(body_pcds.shape[1])[:10000]
                pcd_c = body_pcds[0,iidx]
                iidx2 = torch.randperm(body_pcds.shape[1])[:10000]
                pcd_t = body_pcds[tidx, iidx2].detach()

                # forward pass
                verts_pred = self.deform(verts_c, tidx)
                pcd_pred = self.deform(pcd_c, tidx)

                loss = 0

                # l1 guidance
                loss_guidance = self.opt['guide_loss_weight']*F.l1_loss(verts_t, verts_pred)
                loss += loss_guidance
                info_losses['loss_guide'] += loss_guidance.item()

                #chamfer
                if self.opt['cham_loss_weight']>0 and ii > 0.25*n_iter:
                    loss_alignment = 0.95**(n_iter-ii-1) *self.opt['cham_loss_weight']*chamfer_distance(
                                                pcd_pred.unsqueeze(0),
                                                pcd_t.unsqueeze(0),)[0]
                    loss += loss_alignment
                    info_losses['loss_cham'] += loss_alignment.item()


                if self.opt['homo_loss_weight']>0:
                    vbatch = verts_c + 0.1*torch.randn_like(verts_c)
                    loss_homo = self.opt['homo_loss_weight']*self.model.homogenous_reg(vbatch, tidx, 
                                                                                       norm='charbonnier')
                    loss += loss_homo
                    info_losses['loss_homo'] += loss_homo.item()


                if self.opt['aiap_loss_weight']>0:
                    loss_aiap = self.opt['aiap_loss_weight']*self.aiap_loss(pcd_c, pcd_pred)
                    loss += loss_aiap
                    info_losses['loss_aiap'] += loss_aiap.item()

                # total_loss += loss.item() / tss.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if ii % 10 == 0 or ii==n_iter-1:
                print(f'[log] iter={ii}')
                for key,val in info_losses.items():
                    print(f'[log] {key}={val/tss.shape[0]:.6f}')
                print()

            scheduler.step()

        # save checkpoint and logs
        output = {'model_state_dict': self.model.state_dict()}
        torch.save(output, f"{outputfolder}/last.pt")






    def eval_registration(self, pcds, pcds_normal, meshsrc, outputfolder, outputmeshes=False):
        self.model.load_state_dict(torch.load(outputfolder+'/last.pt')['model_state_dict'])


        self.model.eval()
        verts_c = meshsrc.verts_packed()
        faces = meshsrc.faces_packed()
        verts_pred = []

        with torch.no_grad():
            tts = torch.arange(0, self.opt['motion_model_opt']['n_frames'])
            for tidx in tts:
                x_ = self.deform(verts_c, tidx)
                verts_pred.append(x_)

        verts_pred = torch.stack(verts_pred)
        meshes_pred = Meshes(verts_pred, faces.repeat(tts.shape[0],1,1))


        # eval and save meshes
        evals = {'chamfer_dist_1e4': [],
                'chamfer_normals':[],
                'edge_lengthes':[],
                'verts_vel':[],
                }

        vis_seq_pytorch3d(pcds, normals_to_rgb(pcds_normal),f'{outputfolder}/gt')

        for tt, mesh in enumerate(meshes_pred):
            ## [NOTICE] due to no ground truth mesh, we sample the same number of points in the scan.
            pred_points, pred_normals = sample_points_from_meshes(mesh, return_normals=True,
                                                                  num_samples=pcds[tt].shape[0])
            l2_chamfer, l2_normals = chamfer_eval(pcds[tt:tt+1], pred_points, pcds_normal[tt:tt+1], 
                                                  pred_normals, verbose=False)
            evals['chamfer_dist_1e4'].append(l2_chamfer)
            evals['chamfer_normals'].append(l2_normals)
            vv = mesh.verts_packed()
            ee = mesh.edges_packed()
            edge_lengthes = vv[ee].diff(dim=1).squeeze().norm(dim=-1)
            evals['edge_lengthes'].append(edge_lengthes)
            evals['verts_vel'].append(vv)

            ## save meshes
            if outputmeshes:
                IO().save_mesh(mesh, f'{outputfolder}/source_deformed_{tt}.ply')
                mesh = get_normals_as_textures(mesh).to(device)
                meshimg = render_mesh(mesh)
                plt.imsave(f"{outputfolder}/deformed_{tt}.png",
                           meshimg.detach().cpu().numpy())


        ## compute metrics
        evals['chamfer_dist_1e4'] = torch.stack(evals['chamfer_dist_1e4']).mean().item()
        evals['chamfer_normals'] = torch.stack(evals['chamfer_normals']).mean().item()
        evals['edge_lengthes'] = torch.stack(evals['edge_lengthes']).std(dim=0).max().item()
        evals['verts_vel'] = torch.stack(evals['verts_vel']).diff(dim=0).norm(dim=-1).std(dim=0).mean().item()

        print('eval:')
        for key, val in evals.items():
            print(f"{key}:  {val:.6f}")

        with open(f"{outputfolder}/evals.yaml", 'w') as file:
            yaml.dump(evals, file)
















