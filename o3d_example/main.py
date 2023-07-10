import open3d as o3d
import time
import numpy as np
import matplotlib.pyplot as plt
import hdbscan

pcd = o3d.io.read_point_cloud("./data_object_velodyne/training/velodyne_pcd/000000.pcd")

# visualization with open3d
# o3d.visualization.draw_geometries([pcd])

#--------------------------------------------------------------------------------------------------------------------------------------#

"""Voxel Grid Downsampling"""
print(f"Points before downsampling: {len(pcd.points)} ")
pcd = pcd.voxel_down_sample(voxel_size=0.05)
print(f"Points after downsampling: {len(pcd.points)}")

o3d.visualization.draw_geometries([pcd])

#--------------------------------------------------------------------------------------------------------------------------------------#

# """Statistical Outlier Removal"""
# pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# inlier_cloud = pcd.select_by_index(inliers)
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
# outlier_cloud.paint_uniform_color([1, 0, 0])
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

#--------------------------------------------------------------------------------------------------------------------------------------#

"""Radius Outlier Removal"""
# pcd, inliers = pcd.remove_radius_outlier(nb_points=20, radius=3)
# inlier_cloud = pcd.select_by_index(inliers)
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
# outlier_cloud.paint_uniform_color([1, 0, 0])
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

#--------------------------------------------------------------------------------------------------------------------------------------#

"""RANSAC (RANdom SAmple Consensus)"""
t1 = time.time()
plane_model, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=100)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
outlier_cloud.paint_uniform_color([1, 0, 0])
t2 = time.time()
print(f"Time to segment points using RANSAC {t2 - t1}")
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

#--------------------------------------------------------------------------------------------------------------------------------------#

"""CLUSTERING WITH DBSCAN"""
t3 = time.time()
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.60, min_points=50, print_progress=False))

max_label = labels.max()
print(f'point cloud has {max_label + 1} clusters')
colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
colors[labels < 0] = 0
outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
t4 = time.time()
print(f'Time to cluster outliers using DBSCAN {t4 - t3}')
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

#--------------------------------------------------------------------------------------------------------------------------------------#

# CLUSTERING WITH HDBSCAN
t3 = time.time()
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, gen_min_span_tree=True)
clusterer.fit(np.array(outlier_cloud.points))
labels = clusterer.labels_

max_label = labels.max()
print(f'point cloud has {max_label + 1} clusters')
colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
colors[labels < 0] = 0
outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
t4 = time.time()
print(f'Time to cluster outliers using HDBSCAN {t4 - t3}')
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

#--------------------------------------------------------------------------------------------------------------------------------------#