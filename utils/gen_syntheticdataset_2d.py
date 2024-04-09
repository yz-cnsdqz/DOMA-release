import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from tqdm import tqdm


def get_pixel_coords(img):

  img_height, img_width, n_channels = img.shape
  y_coords = np.linspace(0, 1, img_height)
  x_coords = np.linspace(0, 1, img_width)

  ys, xs = np.meshgrid(x_coords, y_coords)

  coords = np.stack([ys, xs], axis=2)

  return coords


def load_image_as_2d_points(img_path):
  '''Load the image as a set of 2D coordinates and corresponding color values
  '''

  img = np.asarray(Image.open(img_path))[::2,::2,:] # [512x512x3] image

  img = img / 255

  x = get_pixel_coords(img).reshape([-1, 2])
  xrgb = img.reshape([-1, 3])

  # flip y axis
  x[:,1] = 1-x[:,1]

  # normalize the locations to [-1,1]
  x = x*2-1

  return x, xrgb





def rotation_matrix(angle):
    """Create a rotation matrix for a rotation around the z-axis."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])


def shearing_matrix(sigma):
   """create a 2D shearing matrix along the x axis"""
   mat = np.eye(2)
   mat[0,-1] = sigma
   return mat



if __name__=='__main__':

    # load image as points
    ## following this work: https://colab.research.google.com/drive/1xEts0T6E3BMAVqf7l7eE5zj5uAKjcxcT?usp=sharing#scrollTo=_-uTdkEQBfj2
    img_path = 'data/synthetic2d/Home-Adopt-a-Cat-image-1024x1024.jpg'
    xyz_c, fxyz_c = load_image_as_2d_points(img_path)
    num_points = xyz_c.shape[0] # should be 1024*1024
    num_time_steps = 30

    # Angle of rotation per time step (in radians), e.g., pi/20 radians per step
    angle_step = 2*np.pi / num_time_steps
    translations = np.linspace(-1,1,num_time_steps)
    scales = np.linspace(0.5,1.5,num_time_steps)
    shearings = np.linspace(0.0,1.5,num_time_steps)
    # Create a sequence of points over time
    sequence_of_points = np.zeros((num_time_steps, num_points, 2))

    # Apply the rotation over each time step
    for t in tqdm(range(num_time_steps)):
        # shearing
        shearing_mat = shearing_matrix(shearings[t])
        sequence_of_points[t] = xyz_c.dot(shearing_mat.T)

        # scaling
        # sequence_of_points[t] = xyz_c * scales[t]
        # translations
        # sequence_of_points[t] = xyz_c+translations[t]
        # rotation
        # rotation_mat = rotation_matrix(t * angle_step)
        # sequence_of_points[t] = xyz_c.dot(rotation_mat.T)

    outdict = {}
    idxall = np.random.permutation(num_points)
    outdict['xyzs'] = sequence_of_points
    outdict['colors'] = fxyz_c 
    outdict['idxall'] = idxall

    with open('data/synthetic2d/shearing.pkl', 'wb') as f:
       pickle.dump(outdict, f)


