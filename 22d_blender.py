import bpy

# Clear all mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Load the 3D model (replace 'path_to_your_model' with the actual path)
bpy.ops.import_scene.obj(filepath="table.obj")

# Set render resolution
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 800
bpy.context.scene.render.resolution_y = 600

# Set camera and light parameters here...
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
s = np.sqrt(2)/2
camera_pose = np.array([
    [0.0, -s,   s,   0.3],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  s,   s,   0.35],
    [0.0,  0.0, 0.0, 1.0],
])
bpy.add(camera, pose=camera_pose)

# Render the scene and write it to a file
bpy.ops.render.render(write_still=True)
