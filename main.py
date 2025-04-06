import taichi as ti
import numpy as np

# Initialize Taichi with GPU support
ti.init(arch=ti.gpu)

# Define a 3D grid
grid_size = 20
grid = ti.field(dtype=ti.i32, shape=(grid_size, grid_size, grid_size))

@ti.kernel
def initialize_grid():
    # Initialize the grid with a 25% chance each cell is filled
    for x, y, z in grid:
        grid[x, y, z] = 1 if ti.random(ti.f32) < 0.25 else 0

initialize_grid()

# Screen resolution
width, height = 1000, 800
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))  # RGB pixel buffer

# Camera settings
camera_pos = ti.Vector([0.0, 0.0, -3.0])  # Camera position
cube_min = ti.Vector([-1.0, -1.0, -1.0])  # Minimum bounds of the cube
cube_max = ti.Vector([1.0, 1.0, 1.0])    # Maximum bounds of the cube

# Light settings
light_pos = ti.Vector([0.0, 2.0, -3.0])  # Light position
light_color = ti.Vector([1.0, 1.0, 1.0])  # White light
ambient_light = 0.5  # Ambient light intensity

@ti.func
def compute_lighting(hit_point, normal):
    # Compute diffuse lighting based on the light direction and surface normal
    light_dir = (light_pos - hit_point).normalized()
    diff = max(normal.dot(light_dir), 0.0)
    return ambient_light + diff

@ti.func
def phong_shading(hit_point, normal, view_dir):
    # Compute Phong shading (ambient + diffuse + specular)
    light_dir = (light_pos - hit_point).normalized()
    diff = max(normal.dot(light_dir), 0.0)  # Diffuse component
    reflect_dir = (2 * normal.dot(light_dir) * normal - light_dir).normalized()
    spec = max(view_dir.dot(reflect_dir), 0.0) ** 32  # Specular component
    diffuse = diff * light_color
    specular = spec * light_color
    ambient = ambient_light * ti.Vector([1.0, 0.5, 0.0])  # Orange ambient light
    return ambient + diffuse + specular

@ti.func
def ray_box_intersect(ray_origin, ray_dir, box_min, box_max):
    # Ray-box intersection using slab method
    inv_dir = 1.0 / ray_dir
    tmin = (box_min - ray_origin) * inv_dir
    tmax = (box_max - ray_origin) * inv_dir
    t1 = ti.min(tmin, tmax)
    t2 = ti.max(tmin, tmax)
    t_near = ti.max(t1[0], t1[1], t1[2])  # Closest intersection
    t_far = ti.min(t2[0], t2[1], t2[2])   # Farthest intersection
    hit = t_near <= t_far and t_far >= 0  # Check if ray hits the box
    normal = ti.Vector([0.0, 0.0, 0.0])  # Initialize normal
    if hit:
        # Determine which face of the box was hit
        if t_near == t1[0]:
            normal = ti.Vector([-1.0, 0.0, 0.0]) if inv_dir[0] > 0 else ti.Vector([1.0, 0.0, 0.0])
        elif t_near == t1[1]:
            normal = ti.Vector([0.0, -1.0, 0.0]) if inv_dir[1] > 0 else ti.Vector([0.0, 1.0, 0.0])
        elif t_near == t1[2]:
            normal = ti.Vector([0.0, 0.0, -1.0]) if inv_dir[2] > 0 else ti.Vector([0.0, 0.0, 1.0])
    return hit, t_near, normal

@ti.func
def ray_grid_intersect(ray_origin, ray_dir, grid, grid_size, cube_min, cube_max):
    # Ray-grid intersection to check for hits with active cells
    cell_size = (cube_max - cube_min) / grid_size  # Size of each grid cell
    hit = False
    t_near = 0.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    for x, y, z in ti.ndrange(grid_size, grid_size, grid_size):
        if grid[x, y, z] == 1:  # Check if the cell is active
            cell_min = cube_min + ti.Vector([x, y, z]) * cell_size
            cell_max = cell_min + cell_size
            cell_hit, cell_t_near, cell_normal = ray_box_intersect(ray_origin, ray_dir, cell_min, cell_max)
            if cell_hit and (not hit or cell_t_near < t_near):  # Find the closest hit
                hit = True
                t_near = cell_t_near
                normal = cell_normal
    return hit, t_near, normal

@ti.kernel
def render(camera_pos: ti.types.vector(3, ti.f32)):
    # Render the scene by casting rays from the camera
    for i, j in pixels:
        u = (i + 0.5) / width * 2 - 1  # Normalize pixel x-coordinate
        v = (j + 0.5) / height * 2 - 1  # Normalize pixel y-coordinate
        aspect_ratio = width / height
        u *= aspect_ratio  # Adjust for aspect ratio
        ray_dir = ti.Vector([u, v, 1.0]).normalized()  # Ray direction
        hit, t_near, normal = ray_grid_intersect(camera_pos, ray_dir, grid, grid_size, cube_min, cube_max)
        if hit:
            # If the ray hits, compute the color using Phong shading
            view_dir = -ray_dir.normalized()
            color = phong_shading(camera_pos + t_near * ray_dir, normal, view_dir)
            pixels[i, j] = color
        else:
            # Set background color to black if no hit
            # Use an image of the sky as the background
            sky_color = ti.Vector([0.53, 0.81, 0.92])  # Light blue sky color
            pixels[i, j] = sky_color

# Create a GUI window
gui = ti.GUI("Ray Tracing a Cube", res=(width, height))

# Main rendering loop
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        # Handle keyboard input for camera movement
        if e.key == ti.GUI.ESCAPE:
            gui.running = False  # Exit the program
        elif e.key == 'w':
            camera_pos[2] += 0.1  # Move camera forward
        elif e.key == 's':
            camera_pos[2] -= 0.1  # Move camera backward
        elif e.key == 'a':
            camera_pos[0] -= 0.1  # Move camera left
        elif e.key == 'd':
            camera_pos[0] += 0.1  # Move camera right
    render(camera_pos)  # Render the scene
    gui.set_image(pixels)  # Display the rendered image
    gui.show()  # Update the GUI
