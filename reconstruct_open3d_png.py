# depth map to point cloud conversion using Python and OpenCV
import numpy as np
import cv2
import open3d as o3d 

# Load depth map
file = '/Users/seanmao/Pictures/midas/output/Test001-dpt_swin2_large_384.png'
depth_map = cv2.imread(file, cv2.IMREAD_UNCHANGED)
# Load depth map
depth_map = cv2.imread('depth_map.png', cv2.IMREAD_UNCHANGED)

# Create 3D points
height, width = depth_map.shape
fx, fy = width / 2.0, height / 2.0  # Assume the camera has a center projection
cx, cy = width / 2.0, height / 2.0  # Assume the camera center is at the image center

# Create x, y coordinates
x = np.linspace(0, width - 1, width)
y = np.linspace(0, height - 1, height)
x, y = np.meshgrid(x, y)

# Normalize depth map
depth_map = depth_map / np.max(depth_map)

# Backproject to 3D (in camera coordinates)
X = (x - cx) * depth_map / fx
Y = (y - cy) * depth_map / fy
Z = depth_map

# Stack to create point cloud
points = np.stack((X, Y, Z), axis=-1)

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])
