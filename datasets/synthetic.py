import torch
import os,sys,glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import point_cloud_utils as pcu

import json

# Define a custom Dataset
class SyntheticDataset(Dataset):
    def __init__(self, sequence,
                 device=torch.device('cuda',index=0) if torch.cuda.is_available() else torch.device('cpu')):
        """
        Args:
            data (list or ndarray): List of data items or numpy array containing data.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        # self.datapath = '/home/yzhang/workspaces/ResFields/datasets/DeformingThings4D'
        self.seq = sequence
        self.device = device
        
        # load verts
        # data = np.load(os.path.join(self.datapath, self.seq+'.pkl'), allow_pickle=True)
        data = np.load(sequence, allow_pickle=True)
        ### we have the first 25% as guidance points to train and the remaining 75% for test
        np_train = data.shape[1] // 4
        self.train_pts = torch.tensor(data[:,:np_train]).float().to(device)
        self.train_normals = torch.sigmoid(torch.randn_like(self.train_pts[0])).repeat(data.shape[0],1,1) # random normal
        self.test_pts = torch.tensor(data[:,np_train:]).float().to(device)
        self.test_normals = torch.sigmoid(torch.randn_like(self.test_pts[0])).repeat(data.shape[0],1,1)

        self.nt = self.train_pts.shape[0]


if __name__=='__main__':
    dataset = SyntheticDataset('/home/yzhang/workspaces/dpfplus/data/synthetic_cloud/synthetic_rotation.npy')
