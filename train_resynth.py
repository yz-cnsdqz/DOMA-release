import numpy as np
import torch
import torch.nn.functional as F
import glob, os, sys
import time
import argparse

from datasets.resynth import ResynthSubDataset
from models.dpfs import RegistrationOP




device = torch.device('cuda',index=0) if torch.cuda.is_available() else torch.device('cpu')
datapath = '/mnt/hdd/datasets/ReSynth/subset'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True,default='rp_aaron_posed_002')
    parser.add_argument('--seq', type=str, required=True,default='96_jerseyshort_hips')
    parser.add_argument('--motion_model_name', type=str, required=True)
    parser.add_argument('--aiap_loss_weight', type=float, default=0)
    parser.add_argument('--homo_loss_weight', type=float, default=0)
    args = parser.parse_args()

    dataset = ResynthSubDataset(args.subject,
                             args.seq,
                             datapath=datapath,
                             down_sample=2, # first downsample, then take first 30 frames
                             device=device)

    verts = dataset.body_verts
    xyzs = dataset.body_pcds
    xyzs_normal = dataset.body_pcd_n
    meshsrc = dataset.mesh_src

    # define models
    model_opt = get_model_config(args.motion_model_name,verts,
                                 args.aiap_loss_weight, args.homo_loss_weight)

    ## train and test
    modelname = model_opt['motion_model_name']
    outputfolder = f'output/resynthseq_camready_test/{modelname}/{args.subject}.{args.seq}'
    os.makedirs(outputfolder, exist_ok=True)
    model = RegistrationOP(model_opt).to(device)
    # model.registration(verts,  xyzs, outputfolder)    
    model.eval_registration(xyzs, xyzs_normal, meshsrc, outputfolder=outputfolder, outputmeshes=True)




def get_model_config(motion_model_name, verts, 
                     aiap_loss_weight=0.0, homo_loss_weight=0.0):
    if motion_model_name in ['affinefield4d','transfield4d','se3field4d','scaledse3field4d']:
        model_opt = {
        'motion_model_name': motion_model_name,
        'device': device,
        'n_iter': 2000,
        'lr':1e-4,
        'cham_loss_weight': 1e3,
        'guide_loss_weight':1,
        'aiap_loss_weight': aiap_loss_weight,
        'homo_loss_weight': homo_loss_weight,
        'motion_model_opt': {
            'dpf_opt': {
                'in_features':4, # 3D + t
                'hidden_features': 128,
                'n_hidden_layers': 3,
                'max_lens': verts.shape[0],
                },
            'n_frames': verts.shape[0],
            }
        }
    else:
        raise NotImplementedError

    return model_opt



if __name__=='__main__':
    main()
