import torch
import numpy as np
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PerspectiveCameras,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)


DEVICE = 'cuda'
POINT_RADIUS = 0.006
IMAGE_SIZE = 512
POINTS_PER_PIXEL = 25

import trimesh

def save_ply(xyz, ply_path):
  
  ply = trimesh.points.PointCloud(vertices=xyz)
  _ = ply.export(ply_path)

  return





def normalise_verts(V, V_scale=None, V_center=None):

  # Normalize mesh

  if V_scale is not None and V_center is not None:

    V = V - V_center
    V *= V_scale

  else:

    V_max, _ = torch.max(V, dim=0)
    V_min, _ = torch.min(V, dim=0)
    V_center = (V_max + V_min) / 2.
    V = V - V_center

    # Find the max distance to origin
    max_dist = torch.sqrt(torch.max(torch.sum(V**2, dim=-1)))
    V_scale = (1. / max_dist)
    V *= V_scale

  return V, V_scale, V_center


from matplotlib.colors import hsv_to_rgb
def location_to_rgb(xyz: torch.Tensor) -> torch.Tensor:
  if xyz.ndim==3:
    xyz = xyz[0] # only take the first frame
  
  # Normalize the point cloud to the range [0, 1]
  normalized_point_cloud = normalise_verts(xyz)[0]

  # Map the normalized locations to the HSV color space
  # H: angle around the central axis, S: distance from axis, V: height
  # Assuming x, y, z are in the normalized_point_cloud[:, 0], normalized_point_cloud[:, 1], normalized_point_cloud[:, 2] respectively
  hues = torch.atan2(normalized_point_cloud[:, 1], normalized_point_cloud[:, 0]) / (2 * torch.pi) + 0.5
  saturations = torch.sqrt(normalized_point_cloud[:, 0]**2 + normalized_point_cloud[:, 1]**2)
  values = 0.9*torch.ones_like(normalized_point_cloud[:, 2])

  # Combine H, S, V to get the color
  hsv_colors = torch.stack((hues, saturations, values), dim=1).detach().cpu().numpy()

  # Convert HSV to RGB since matplotlib works with RGB
  rgb_colors = hsv_to_rgb(hsv_colors)
  
  return torch.tensor(rgb_colors).clamp(0,1)





def normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
  """
  Convert mesh normals to RGB color representation.

  Args:
      normals (torch.Tensor): Mesh normals.

  Returns:
      torch.Tensor: RGB colors based on the normals.
  """
  return torch.abs(normals * 0.5 + 0.5)


def get_point_renderer(image_size, radius=0.05, points_per_pixel=50):

  raster_settings = PointsRasterizationSettings(
      image_size=image_size,
      radius = radius,
      points_per_pixel = points_per_pixel
      )

  rasterizer = PointsRasterizer(cameras=FoVOrthographicCameras(),
                                raster_settings=raster_settings)
  renderer = PointsRenderer(
      rasterizer=rasterizer,
      compositor=AlphaCompositor(background_color=(0, 0, 0))
  )

  return renderer

def get_camera(dist=1, elev=0, azim=0):

  R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)

  cam = PerspectiveCameras(R=R, T=T)

  return cam

def render_points(x, xf, dist=1.0, elev=0, azim=0,
                  image_size=IMAGE_SIZE,
                  radius=POINT_RADIUS,
                  points_per_pixel=POINTS_PER_PIXEL, scale_val=0.22,
                  device=DEVICE):
  """point rendering for visualization.
  change azim, dist to adjusting the view angle;
  change scale_val to zoom-in /zoom-out
  """
  x = x.to(device)
  xf = xf.to(device)
  renderer = get_point_renderer(image_size=image_size, radius=radius, points_per_pixel=points_per_pixel)
  R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
  cam = FoVOrthographicCameras(R=R, T=T, scale_xyz=((scale_val, scale_val, scale_val),)).to(device)

  pcl = Pointclouds(points=x.unsqueeze(0), features=xf.unsqueeze(0)).to(device)

  img = renderer(pcl, cameras=cam)[0]

  return img



import matplotlib.pyplot as plt
def vis_seq_pytorch3d(pcds, rgbs, prefix):
  # pcds = pcds - pcds.mean(dim=0, keepdim=True)

  if pcds.ndim ==3 and rgbs.ndim==2:
    for tidx, pcd in enumerate(pcds):
        img = render_points(pcd, rgbs)
        plt.imsave(f"{prefix}_frame_{tidx}.png", img.detach().cpu().numpy())
  elif pcds.ndim==2 and rgbs.ndim==3:
    for tidx, rgb in enumerate(rgbs):
      img = render_points(pcds, rgb)
      plt.imsave(f"{prefix}_frame_{tidx}.png", img.detach().cpu().numpy())
  elif pcds.ndim==3 and rgbs.ndim==3:
    for tidx, rgb in enumerate(rgbs):
      img = render_points(pcds[tidx], rgb)
      plt.imsave(f"{prefix}_frame_{tidx}.png", img.detach().cpu().numpy())



import matplotlib.pyplot as plt  
def vis_seq_2d(pcds, rgbs, prefix):
  pcds = pcds.detach().cpu().numpy()
  rgbs = rgbs.detach().cpu().numpy()
  px = 1/plt.rcParams['figure.dpi']
  for tidx, rgb in enumerate(rgbs):
    xyz = pcds[tidx]
    rgb = rgb
    plt.figure(figsize=(512*px, 512*px))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.scatter(xyz[:,0], xyz[:,1], c=rgb, s=0.02)
    # Make the axes invisible
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # Save the plot to a file
    plt.savefig(f"{prefix}_frame_{tidx}.png",
                bbox_inches='tight', pad_inches=0)
    plt.clf()






from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    Textures,
    look_at_view_transform,
    FoVOrthographicCameras,
    Materials,
    RasterizationSettings,
    BlendParams,
    MeshRenderer,
    MeshRasterizer,
    AmbientLights,
    HardPhongShader,
)


def get_normals_as_textures(mesh):
    """
    Create textures from mesh normals.

    Args:
        mesh (Meshes): Input mesh.

    Returns:
        Meshes: New mesh with normals as textures.
    """
    normals = mesh.verts_normals_packed()
    textures = Textures(verts_rgb=normals_to_rgb(normals).unsqueeze(0))
    return Meshes(mesh.verts_packed().unsqueeze(0), mesh.faces_packed().unsqueeze(0), textures)





def create_mesh_renderer(cameras, image_size=IMAGE_SIZE, device='cuda'):
    """
    Create a mesh renderer.

    Args:
        cameras (FoVOrthographicCameras): Camera setup.
        image_size (int, optional): Image size. Defaults to DEFAULT_IMAGE_SIZE.
        device (str, optional): Device for computation. Defaults to 'cuda'.

    Returns:
        MeshRenderer: Mesh renderer.
    """
    materials = Materials(
        device=device,
        specular_color=[[0.0, 0.0, 0.0]],
        shininess=0.0
    )

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        cull_backfaces=True
    )

    blend_params = BlendParams(background_color=(0, 0, 0))

    lights = AmbientLights(ambient_color=(1, 1, 1), device=device)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),

        shader=HardPhongShader(device=device,
                               cameras=cameras,
                               blend_params=blend_params,
                               lights=lights,
                               materials=materials)
    )

    return renderer

def render_mesh(mesh, dist=1, elev=0, azim=0,
                image_size=IMAGE_SIZE, radius=0.01, scale_val=0.85,
                device=DEVICE):
    """
    Render a mesh.

    Args:
        mesh (Meshes): Input mesh.
        dist (float, optional): Distance from the camera. Defaults to 1.
        elev (float, optional): Elevation angle. Defaults to 0.
        azim (float, optional): Azimuth angle. Defaults to 0.
        image_size (int, optional): Image size. Defaults to DEFAULT_IMAGE_SIZE.
        radius (float, optional): Radius. Defaults to 0.01.
        scale_val (float, optional): Scaling value. Defaults to 1.0.
        device (str, optional): Device to use for rendering. Defaults to DEVICE.

    Returns:
        torch.Tensor: Rendered image.
    """
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cam = FoVOrthographicCameras(R=R, T=T, scale_xyz=((scale_val, scale_val, scale_val),)).to(device)

    renderer = create_mesh_renderer(cam, image_size=image_size)

    img = renderer(mesh, cameras=cam)[0]

    return img