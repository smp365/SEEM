import pyrender
import numpy as np
import trimesh

# Load the 3D model
mesh = trimesh.load('path_to_your_3d_model.obj')

# Create a scene
scene = pyrender.Scene()

# Check if the loaded object is a scene (i.e., contains multiple meshes)
if isinstance(loaded, trimesh.Scene):
    # Add each mesh in the scene to the pyrender scene
    for mesh in loaded.geometry.values():
        scene.add(pyrender.Mesh.from_trimesh(mesh))
else:
    # Add the single mesh to the pyrender scene
    scene.add(pyrender.Mesh.from_trimesh(loaded))

# Set the camera parameters
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
s = np.sqrt(2)/2
camera_pose = np.array([
    [0.0, -s,   s,   0.3],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  s,   s,   0.35],
    [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)

# Render the scene
renderer = pyrender.OffscreenRenderer(640, 480)
color, depth = renderer.render(scene)

# Save the rendered image
pyrender.image.save_image('rendered_image.png', color, flip_up_down=True)
