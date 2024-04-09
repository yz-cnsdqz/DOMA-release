import torch
import os,sys,glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import point_cloud_utils as pcu
from pytorch3d.io import load_objs_as_meshes


class ResynthSubDataset(Dataset):
    def __init__(self, subject='rp_aaron_posed_002', 
                 sequence='96_jerseyshort_hips',
                 datapath = '/mnt/hdd/datasets/ReSynth/subset',
                 down_sample=2,
                 n_frames = 30,
                 device=torch.device('cuda',index=0) if torch.cuda.is_available() else torch.device('cpu')):
        """
        Args:
            data (list or ndarray): List of data items or numpy array containing data.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.sub = subject
        self.seq = sequence
        self.device = device
        self.datapath =datapath

        # load verts
        self.body_verts, self.body_pcds, self.body_pcd_n = self.load_seq(down_sample)
        self.body_verts = self.body_verts[:n_frames]
        self.body_pcds = self.body_pcds[:n_frames]
        self.body_pcd_n = self.body_pcd_n[:n_frames]

        self.nt, self.nb = self.body_verts.shape[:2]

        # load mesh
        meshfile = os.path.join(self.datapath, f"{self.sub}.{self.seq}.00020.obj")
        self.mesh_src = load_objs_as_meshes([meshfile], device=device)
                
    # load data
    def load_seq(self, down_sample):
        body_verts = []
        scan_pc = []
        scan_normal = []
        files = sorted(glob.glob(
            os.path.join(self.datapath, f"{self.sub}.{self.seq}.*.npz")))
        for file in files[::down_sample]:
            data = np.load(file)
            body_verts.append(data['body_verts'])
            scan_pc.append(data['scan_pc'])
            scan_normal.append(data['scan_n'])
        body_verts = torch.tensor(np.stack(body_verts)).float().to(self.device)
        scan_pc = torch.tensor(np.stack(scan_pc)).float().to(self.device)
        scan_normal = torch.tensor(np.stack(scan_normal)).float().to(self.device)


        return body_verts, scan_pc, scan_normal
