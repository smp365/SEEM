import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44, Vector3
import trimesh

# Load the 3D model
mesh = trimesh.load_mesh('desk.obj')

# Create a standalone ModernGL context
ctx = moderngl.create_standalone_context()

# Create a framebuffer with a texture attached
fbo = ctx.framebuffer(color_attachments=[ctx.texture((512, 512), 4)])

# Create a shader
prog = ctx.program(
    vertex_shader='''
    #version 330

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 proj;

    in vec3 in_position;

    void main() {
        gl_Position = proj * view * model * vec4(in_position, 1.0);
    }
    ''',
    fragment_shader='''
    #version 330

    out vec4 fragColor;

    void main() {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);  # Red color
    }
    '''
)

# Set up a projection and view matrix
proj = Matrix44.perspective_projection(45.0, 1.0, 0.1, 100.0)
view = Matrix44.look_at(Vector3([3, 3, 3]), Vector3([0, 0, 0]), Vector3([0, 1, 0]))

# Create a vertex buffer and an element buffer
vbo = ctx.buffer(mesh.vertices.astype('f4').tobytes())
ebo = ctx.buffer(mesh.faces.astype('i4').tobytes())

# Create a vertex array object
vao = ctx.simple_vertex_array(prog, vbo, 'in_position', index_buffer=ebo)

# Use the framebuffer
fbo.use()

# Clear the framebuffer
ctx.clear()

# Draw the model
prog['model'].write(Matrix44.identity().astype('f4').tobytes())
prog['view'].write(view.astype('f4').tobytes())
prog['proj'].write(proj.astype('f4').tobytes())
vao.render(moderngl.TRIANGLES)

# Read the pixels back
pixels = np.frombuffer(fbo.read(), dtype='f4')

# Reshape and flip the image vertically (OpenGL's origin is bottom-left)
pixels = np.reshape(pixels, (512, 512, 4))[::-1, :, :]

# Convert to 8-bit per channel
pixels = (pixels * 255).astype('u1')

# Create a PIL image and save it
Image.fromarray(pixels).save('output.png')
