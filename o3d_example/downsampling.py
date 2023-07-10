import open3d as o3d

pcd = o3d.io.read_point_cloud("./data_object_velodyne/training/velodyne_pcd/000000.pcd")

print(f"Points before downsampling: {len(pcd.points)} ")
# Points before downsampling: 115384 
pcd = pcd.voxel_down_sample(voxel_size=0.2)
print(f"Points after downsampling: {len(pcd.points)}")
# Points after downsampling: 22625

# visualization with open3d
o3d.visualization.draw_geometries([pcd])