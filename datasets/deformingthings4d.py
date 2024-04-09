import torch
import os,sys,glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import point_cloud_utils as pcu
from pytorch3d.io import load_ply
from pytorch3d.structures.meshes import Meshes



import json

# Define a custom Dataset
class DeformingThings4DDataset(Dataset):
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
        self.train_pts = torch.tensor(data['train_pts']).to(device)
        self.train_normals = torch.tensor(data['train_normals']).to(device)
        self.test_pts = torch.tensor(data['test_pts']).to(device)
        self.test_normals = torch.tensor(data['test_normals']).to(device)

        self.nt = self.train_pts.shape[0]


        # load mesh ply files and put to a single pytorch3d Mesh object
        verts = []
        faces = []
        meshdir = sequence.replace('.pkl', '_meshes')
        n_meshes = min(100, len(glob.glob(meshdir+'/*.ply')))

        for t in range(n_meshes):
            verts_, faces_ = load_ply(meshdir+f'/{t:04d}.ply')
            verts.append(verts_)
            faces.append(faces_)
        
        self.meshes = Meshes(verts=verts, faces=faces).to(device)




if __name__=='__main__':
    dataset = DeformingThings4DDataset('/home/yzhang/workspaces/ResFields/datasets/DeformingThings4D/bear3EP_Agression.anime.pkl')
