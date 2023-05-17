# depth map to point cloud conversion using Python and OpenCV
import numpy as np
# doesn't support amazon linux 2, use plyfile instead
# import open3d as o3d 
from plyfile import PlyData, PlyElement
#from midas.pfm import read_pfm

# read PFM files
def read_pfm(file_path):
    with open(file_path, 'rb') as file:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = file.readline().decode('latin-1')
        if 'PF' in type:
            channels = 3
        elif 'Pf' in type:
            channels = 1
        else:
            raise Exception('Not a PFM file.')

        # Line 2: width height
        line = file.readline().decode('latin-1')
        width, height = map(int, line.split())

        # Line 3: scale
        scale = float(file.readline().decode('latin-1'))

        # Following lines: data (width*height*channels floats), in row-major order,
        # i.e., the same ordering as produced by:
        # np.dstack([im.astype(float) for im in cv2.split(img)]).tofile(f)
        data = np.fromfile(file, dtype=np.float32, count=height*width*channels)
        data = np.reshape(data, (height, width, channels))
        data = np.flip(data, axis=0)
        # If grayscale, return a 2D array instead of a 3D array
        if channels == 1:
            data = data[:, :, 0]

        return data, scale

# Load depth map
depth_map, scale = read_pfm('/home/ec2-user/midas/MiDaS/output/Test001-dpt_swin2_large_384.pfm')

# Create 3D points
height, width = depth_map.shape
fx, fy = width / 2.0, height / 2.0  # Assume the camera has a center projection
cx, cy = width / 2.0, height / 2.0  # Assume the camera center is at the image center

# Create x, y coordinates
x = np.linspace(0, width - 1, width)
y = np.linspace(0, height - 1, height)
x, y = np.meshgrid(x, y)

# Backproject to 3D (in camera coordinates)
X = (x - cx) * depth_map / fx
Y = (y - cy) * depth_map / fy
Z = depth_map

# Stack to create point cloud
vertex = np.dstack([X, Y, Z]).reshape(-1, 3)

# Convert to structured array
vertex = np.core.records.fromarrays(vertex.transpose(), names='x, y, z', formats='f4, f4, f4')

# Create ply element
vertex_element = PlyElement.describe(vertex, 'vertex')

# Create ply data
ply_data = PlyData([vertex_element])

# Write ply data to file
ply_data.write('Test001_point_cloud.ply')
