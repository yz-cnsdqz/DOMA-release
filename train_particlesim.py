import numpy as np
import torch
import torch.nn.functional as F
import glob, os, sys
import argparse
import time

from datasets.particlesim import ParticlesimDataset
from models.dpfs import MotionField




device = torch.device('cuda',index=0) if torch.cuda.is_available() else torch.device('cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence',  required=True)
    parser.add_argument('--start',  type=int, default=100)
    parser.add_argument('--end',  type=int, default=120)
    parser.add_argument('--motion_model_name',  required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_printoptions(sci_mode=False)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    
    seq = args.sequence
    
    dataset = ParticlesimDataset(sequence=seq,
                                 start=args.start,
                                 end = args.end,
                                 device=device)

    train_pts = dataset.train_pts
    test_pts = dataset.test_pts
    test_normals = torch.rand_like(test_pts)[0]
    test_normals = test_normals.repeat(test_pts.shape[0],1,1)


    # define models
    model_opt_all = {
        'transfield4d':{
            'device': device,
            'motion_model_opt': {
                'dpf_opt': {
                    'in_features':4, # 3D + t
                    'out_features':3, # rotcont + 3D transl
                    'hidden_features': 64,
                    'n_hidden_layers': 2,
                    'max_lens': train_pts.shape[0],
                    },
                'n_frames': train_pts.shape[0],
                'homo_loss_weight': 0.0,
                'elastic_loss_weight': 0.0,
                'n_iter': 5000,
                'lr':1e-4,
            }},

        
        'dpfbag':{
            'device': device,
            'motion_model_opt': {
                'dpf_opt': {
                    'in_features':3, # 3D + t
                    'out_features':3, # rotcont + 3D transl
                    'hidden_features': 64,
                    'n_hidden_layers': 2,
                    'max_lens': train_pts.shape[0],
                    },
                'n_frames': train_pts.shape[0],
                'homo_loss_weight': 0.0,
                'elastic_loss_weight': 0.0,
                'n_iter': 5000,
                'lr':1e-4,
            }},
            
        'affinefield4d':{
            'device': device,
            'motion_model_opt': {
                'dpf_opt': {
                    'in_features':4, # 3D + t
                    'out_features':12, # rotcont + 3D transl
                    'hidden_features': 64,
                    'n_hidden_layers': 2,
                    'max_lens': train_pts.shape[0],
                    },
                'n_frames': train_pts.shape[0],
                'homo_loss_weight': 0.0,
                'elastic_loss_weight': 0.0,
                'n_iter': 5000,
                'lr':1e-4,
            }}
    }


    model_opt = model_opt_all[args.motion_model_name]
    model_opt['motion_model_name'] = args.motion_model_name

    model = MotionField(model_opt).to(device)
    
    homow = model_opt['motion_model_opt']['homo_loss_weight']
    outputfolder = f'output/particlesim/start{args.start}_end{args.end}/{args.motion_model_name}_homo{homow}/{os.path.basename(seq)}'
    os.makedirs(outputfolder, exist_ok=True)
    model.train_motion_field(train_pts, outputfolder)
    model.test_motion_field(test_pts, test_normals, outputfolder)
    


if __name__=='__main__':
    main()
    
    # one could have a separate script to run    
    # models = ['affinefield4d', 'transfield4d', 'dpfbag']

    # t0 = 10
    # nt = 350

    # while t0 < nt:
    #     for model in models:
    #         cmd = f"python train_particlesim.py --sequence=/mnt/hdd/datasets/doma_datasets/particlesim/particles_0000.npy --motion_model_name={model} --start={t0} --end={t0+10}"
    #         subprocess.call(cmd, shell=True)

    #     t0 += 15
    






