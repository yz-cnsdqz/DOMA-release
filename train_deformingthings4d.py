import numpy as np
import torch
import glob, os, sys
import argparse


from datasets.deformingthings4d import DeformingThings4DDataset
from models.dpfs import MotionField



device = torch.device('cuda',index=0) if torch.cuda.is_available() else torch.device('cpu')
datapath = '/home/yzhang/workspaces/ResFields/datasets/DeformingThings4D'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_model_name', required=True, help='type the name of the motion model.')
    parser.add_argument('--eval_mesh_alignment', action='store_true', default=False)
    args = parser.parse_args()

    sequences = sorted(glob.glob(f'{datapath}/*.pkl'))

    for seq in sequences: 
        print()
        print(f'--seq: {seq}')
        # load data
        dataset = DeformingThings4DDataset(sequence=seq,device=device)
        train_pts = dataset.train_pts[:100]
        train_normals = dataset.train_normals[:100]
        test_pts = dataset.test_pts[:100]
        test_normals = dataset.test_normals[:100]
        meshes = dataset.meshes

        # define models
        model_opt = get_model_config(args.motion_model_name, 
                                     train_pts)
        model = MotionField(model_opt).to(device)
        
        
        # train test op
        outputfolder = f'output/motionprediction_deformingthings4d/{args.motion_model_name}/{os.path.basename(seq)}'
        os.makedirs(outputfolder, exist_ok=True)
        model.train_motion_field(train_pts, outputfolder)
        model.test_motion_field(test_pts, test_normals, outputfolder)
        
        # especially for mesh deformation
        if args.eval_mesh_alignment:
            model.eval_mesh_deformation(meshes, outputfolder)
    



def get_model_config(motion_model_name, train_pts):
    if motion_model_name in ['affinefield4d','transfield4d','se3field4d','scaledse3field4d']:
        model_opt = {
                'motion_model_name': motion_model_name,
                'device': device,
                'motion_model_opt': {
                    'dpf_opt': {
                        'in_features':4, # 3D + t
                        'hidden_features': 128,
                        'n_hidden_layers': 2,
                        'max_lens': train_pts.shape[0],
                        },
                    'n_frames': train_pts.shape[0],
                    'homo_loss_weight': 0,
                    'elastic_loss_weight': 0,
                    'n_iter': 1000,
                    'lr':1e-4
                }
            }
    elif motion_model_name=='banmo':
        model_opt = {
            'motion_model_name': motion_model_name,
            'device': device,
            'motion_model_opt': {
                'n_bones': 25,
                'n_freqs': 10,
                't_embed_dim':128,
                'n_frames': train_pts.shape[0],
                'xyz_min': torch.amin(train_pts[0],dim=0).detach().cpu(),
                'xyz_max': torch.amax(train_pts[0],dim=0).detach().cpu(),
                'optimize_bone_locs': True,
                'n_iter': 1000,
                'lr':1e-3
            }
        }
    elif motion_model_name=='bonecloud':
        model_opt = {
            'motion_model_name': motion_model_name,
            'device': device,
            'motion_model_opt': {
                'n_bones': 1024,
                'n_frames': train_pts.shape[0],
                'bone_sample': 'rand',
                'sigma': 10.,
                'xyz_min': torch.amin(train_pts[0],dim=0).detach().cpu(),
                'xyz_max': torch.amax(train_pts[0],dim=0).detach().cpu(),
                'optimize_bone_locs': True,
                'n_iter': 1000,
                'lr':1e-3
            }
        }
    else:
        raise NotImplementedError

    return model_opt



if __name__=='__main__':
    main()

