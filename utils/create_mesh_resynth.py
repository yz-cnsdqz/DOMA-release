import open3d as o3d
import numpy as np


import os, glob

files = sorted(glob.glob('/mnt/hdd/datasets/ReSynth/createdmeshes/rp_eric*.npz'))

for file in files:
    print(f'process: {file}')
    data = np.load(file)
    points = data['scan_pc']
    normals = data['scan_n']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # vertices_to_remove = densities < np.quantile(densities, 0.0005)
    # mesh.remove_vertices_by_mask(vertices_to_remove)


    # Calculate the normals of the vertex
    mesh.compute_vertex_normals()
    # Paint it gray. Not necessary but the reflection of lighting is hardly perceivable with black surfaces.
    mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))

    o3d.io.write_triangle_mesh(file.replace('.npz', '.obj'), mesh)
