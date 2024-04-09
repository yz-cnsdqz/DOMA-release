import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from scipy.stats import sem
import point_cloud_utils as pcu

device = 'cuda'

def chamfer_eval(x, y, xn, yn, verbose=False):
    """
    Compute the chamfer distance and cosine normals between two point sets.

    Args:
        x (torch.Tensor): The first set of points (B x N x 3).
        y (torch.Tensor): The second set of points (B x M x 3).
        xn (torch.Tensor): The normals of the first set of points (B x N x 3).
        yn (torch.Tensor): The normals of the second set of points (B x M x 3).
        verbose (bool): If True, print the computed chamfer distances and cosine normals.

    Returns:
        torch.Tensor: The chamfer distance between x and y (B).
        torch.Tensor: The cosine normals between xn and yn (B).
    """
    l2_chamfer, l2_normals = chamfer_distance(x, y, x_normals=xn, y_normals=yn, norm=2)
    l2_chamfer, l2_normals = (10**4) * l2_chamfer, l2_normals

    if verbose:
        print("Chamfer L2 (x 10^-4): %f, Chamfer Cosine Normals: %f" %
              (float(l2_chamfer), float(l2_normals)))

    return l2_chamfer, l2_normals

def eval_surface_reconstruction_mesh(pred_mesh, gt_mesh, n_samples=10**6, device='cuda'):
    """
    Evaluate the surface reconstruction performance for two meshes.

    Args:
        pred_mesh_path (str): The file path to the predicted mesh.
        gt_mesh_path (str): The file path to the ground truth mesh.
        n_samples (int): The number of points to sample from each mesh. Defaults to 10^6.
        device (str): The device to perform computations on. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The chamfer distance between predicted and ground truth meshes.
        torch.Tensor: The cosine normals between predicted and ground truth meshes.
    """
    
    gt_points, gt_normals = sample_points_from_meshes(gt_mesh, return_normals=True, num_samples=n_samples)
    pred_points, pred_normals = sample_points_from_meshes(pred_mesh, return_normals=True, num_samples=n_samples)

    l2_chamfer, l2_normals = chamfer_eval(gt_points, pred_points, gt_normals, pred_normals, verbose=True)

    return l2_chamfer, l2_normals

# the code taken from: https://github.com/rabbityl/DeformationPyramid
def scene_flow_metrics(pred, labels, strict=0.025, relax = 0.05):

    l2_norm = torch.sqrt(torch.sum((pred - labels) ** 2, 1)).cpu()  # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, 1)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: AccS
    error_lt_5 = torch.BoolTensor((l2_norm < strict))
    relative_err_lt_5 = torch.BoolTensor((relative_err < strict))
    AccS = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: AccR
    error_lt_10 = torch.BoolTensor((l2_norm < relax))
    relative_err_lt_10 = torch.BoolTensor((relative_err < relax))
    AccR = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    # NOTE: outliers
    relative_err_lt_30 = torch.BoolTensor(relative_err > 0.3)
    outlier = torch.mean(relative_err_lt_30.float()).item()

    return EPE3D, AccS*100, AccR*100 #, outlier*100


def eval_scene_flow(pred_mesh_path, gt_mesh_path, strict=0.025, relax = 0.05):

    vpred = torch.Tensor(pcu.load_mesh_v(pred_mesh_path))
    vgt = torch.Tensor(pcu.load_mesh_v(gt_mesh_path))

    epe, accs, accr =  scene_flow_metrics(vpred, vgt)

    return epe, accs, accr