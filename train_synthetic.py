import numpy as np
import torch
import torch.nn.functional as F
import glob, os, sys
import argparse


from datasets.synthetic import SyntheticDataset
from models.dpfs import MotionField



device = torch.device('cuda',index=0) if torch.cuda.is_available() else torch.device('cpu')
datapath = '/home/yzhang/workspaces/dpfplus/data/synthetic'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_model_name', type=str, required=True)
    parser.add_argument('--elastic_loss_weight', type=float, default=0)
    parser.add_argument('--homo_loss_weight', type=float, default=0)
    parser.add_argument('--expid', type=str, default='')
    args = parser.parse_args()

    if args.expid=='':
        exp_id = f"{np.random.randint(1,10000):05d}"
    else:
        exp_id = args.expid

    sequences = sorted(glob.glob(f'{datapath}/*.npy'))

    for seq in sequences:
        print()
        print(f'--seq: {seq}')
        # load data
        dataset = SyntheticDataset(sequence=seq,device=device)
        train_pts = dataset.train_pts
        train_normals = dataset.train_normals
        test_pts = dataset.test_pts
        test_normals = dataset.test_normals

        # define models
        model_opt = get_model_config(args.motion_model_name, 
                                     train_pts,
                                     args.elastic_loss_weight,
                                     args.homo_loss_weight)
        model = MotionField(model_opt).to(device)
        

        # train test op
        outputfolder = f'output/motionprediction_synthetic_{exp_id}/{args.motion_model_name}/{os.path.basename(seq)}'
        os.makedirs(outputfolder, exist_ok=True)

        # set timers
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        model.train_motion_field(train_pts, outputfolder)
        model.test_motion_field(test_pts, test_normals, outputfolder)
        
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        total_time = start.elapsed_time(end)
        
        print(f"Total training time: {1e-3*total_time} seconds")



def get_model_config(motion_model_name, train_pts, elastic_loss_weight, homo_loss_weight):
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
                    'homo_loss_weight': homo_loss_weight,
                    'elastic_loss_weight': elastic_loss_weight,
                    'n_iter': 1000,
                    'lr':1e-4
                }
            }
    else:
        raise NotImplementedError

    return model_opt



if __name__=='__main__':
    main()