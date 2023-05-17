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
points = np.stack((X, Y, Z), axis=-1)

# Create Open3D point cloud object
#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))

# Visualize point cloud
# o3d.visualization.draw_geometries([pcd])

# Save point cloud to a PLY file
#o3d.io.write_point_cloud("Test001_point_cloud.ply", pcd)

# Reshape the points into a 2D array where each row is a point
points_2d = points.reshape(-1, 3)

# Create a PlyElement instance for the points
vertex = np.array(points_2d, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

# Flatten the vertex array into a 1D array of tuples 
vertex = np.dstack([X, Y, Z]).reshape(-1, 3)
vertex_element = PlyElement.describe(vertex, 'vertex')

# Write the PLY file
PlyData([vertex_element]).write('output.ply')
