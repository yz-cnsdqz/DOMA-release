# Degrees of Freedom Matter: Inferring Dynamics from Point Trajectories


[page](https://yz-cnsdqz.github.io/eigenmotion/DOMA/), [paper]()


## License

The code is distributed under the [MIT License](https://opensource.org/license/mit). 


## Install
Clone repository, create an environment with Python 3.10 (e.g. with conda) and install dependencies in `requirements_minimal.txt` via pip:

```bash
git clone https://github.com/yz-cnsdqz/DOMA-release.git
cd DOMA-release

# Create environment (skip this if you have it)
conda create -n doma python=3.10 -y
conda activate doma

# Install gcc and g++ (if not available in your system)
conda install -c conda-forge gcc gxx -y 

# Install dependencies
pip install -r requirements_minimal.txt
```

**Note:** The `requirements_minimal.txt` contains only the packages actually used by this codebase. The original `requirements.txt` includes many additional packages that are not necessary for running DOMA.

Essential python libraries: 
```
torch==2.0.1
pytorch3d==0.7.4
torchgeometry==0.1.2
numpy==1.23.1
trimesh==3.23.5
PyYAML==6.0.1
```

## Data

### DeformingThings4D

We exploited this dataset for novel point motion prediction. Before downloading this dataset, one should check its [official site](https://github.com/rabbityl/DeformingThings4D) and agree its term of use.
Please see our extended [ResField](https://drive.google.com/file/d/1234X1lywiOr0j90JVBYyvlv7CKmX77Kn/view?usp=sharing) for how we processed this dataset.

### Synthetic

We created this dataset with 4 sequences for novel point motion prediction. One can download our version [here](https://drive.google.com/drive/folders/1Gll1IDqlGM1BunmOwvgXd5JPrOlS9pTk?usp=sharing), or generate new synthetic data via `utils/gen_syntheticdataset_3d.py`.

### Resynth
We exploited this dataset for temporal mesh alignment with guidance.
Please see its [official website](https://pop.is.tue.mpg.de) for more information and downloading.

In our experiment, we choose 16 sequences from 4 subjects in the `packed` sequences in the test split.
For each sequence, we first perform down-sampling by every 2 frames, and then select the first 30 frames as the new sequence to model. The employed sequences are present in our paper supp. mat.

### Particle Simulation
We conducted an additional experiment in the supp. mat, and leveraged [this github repo](https://github.com/SebLague/Fluid-Sim) to produce the data. Running and extracting the simulated particles require some basic knowledge of Unity3D.
Alternatively, one can download our extracted and processed data [here](https://drive.google.com/drive/folders/1Kmgbd1R-KaziAkqLx6LFmorBMTPNdcf6?usp=sharing).

## Usage
The training, evaluation, and rendering scripts are in these `train_*.py` files.

### Novel Point Motion Prediction
Besides the three baselines implemented in our modified [ResField repo](), one can run set up the dataset path, and run e.g.
```
python train_deformingthings4d.py --motion_model_name=affinefield4d 
```
All options of `motion_model_name` are listed in the `MotionField` class in `models/dpfs.py`. 

The results are saved into the `output` folder. One can modify the point renderer settings in `render_points()` of `utils/vis.py`. Details of the configurations are in `train_deformingthings4d.py`.

Similarly, one can use the following to learn motion fields for the 4 synthetic sequences.
```
python train_synthetic.py --motion_model_name=affinefield4d --homo_loss_weight=0.1
```

Run the following one to learn the motion field in the simulated fluid field.
```
python train_particlesim.py --sequence=/doma_datasets/particlesim/particles_0000.npy --motion_model_name=transfield4d --start=100 --end=110
```

### Temporal Mesh Alignment with Guidance

One can start with `train_resynth.py`, e.g.
```
python train_resynth.py --subject=rp_aaron_posed_002 --seq=96_jerseyshort_hips --motion_model_name=affinefield4d
```


## Citation
```
@inproceedings{DOMA,
    title   = {Degrees of Freedom Matter: Inferring Dynamics from Point Trajectories},
    author  = {Yan Zhang and Sergey Prokudin and Marko Mihajlovic and Qianli Ma and Siyu Tang},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year    = {2024}
}
```

### Related Projects
```
@inproceedings{mihajlovic2024ResFields,
   title={{ResFields}: Residual Neural Fields for Spatiotemporal Signals},
   author={Mihajlovic, Marko and Prokudin, Sergey and Pollefeys, Marc and Tang, Siyu},
   booktitle={International Conference on Learning Representations (ICLR)},
   year={2024}
} 

@inproceedings{prokudin2023dynamic,
    title = {Dynamic Point Fields},
    author = {Prokudin, Sergey and Ma, Qianli and Raafat, Maxime and Valentin, Julien and Tang, Siyu},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    pages = {7964--7976},
    month = oct,
    year = {2023}
}
```
















