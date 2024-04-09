import torch
import os,sys,glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import point_cloud_utils as pcu

import json

# Define a custom Dataset
class ParticlesimDataset(Dataset):
    def __init__(self, sequence,
                 start = 100,
                 end = 120,
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
        data = np.load(sequence, allow_pickle=True)[::2][start:end]  #downsample
        data = torch.tensor(data).float().to(device)
        self.nt, nb = data.shape[:2]
        indices = torch.randperm(data.shape[1])
        
        self.train_pts = data[:,indices[:nb//2]]
        self.test_pts = data[:,indices[nb//2:]]
        
        
