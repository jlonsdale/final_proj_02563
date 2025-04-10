import taichi as ti

ti.init(arch=ti.cpu)  # Initialize Taichi, you can change to ti.gpu if needed

@ti.dataclass
class Ray:
    origin: ti.types.vector(3, ti.f32)  # 3D vector of floats
    direction: ti.types.vector(3, ti.f32)

@ti.dataclass
class Hit:
    point: ti.types.vector(3, ti.f32)  # 3D vector of floats
    normal: ti.types.vector(3, ti.f32)  # 3D vector of floats
    distance: ti.f32
    material: ti.i32  # Integer material identifier (optional, can be -1 for None)

# Define a 3D grid
grid_size = 10
grid = ti.field(dtype=ti.i32, shape=(grid_size, grid_size, grid_size))
cube_centers = ti.Vector.field(3, dtype=ti.f32, shape=(grid_size**3,))
cube_materials = ti.field(dtype=ti.i32, shape=(grid_size**3,))
light_dir = ti.Vector([0.0, -1.0, 0.0]) 


@ti.kernel
def initialize_grid():
    # Initialize the grid with a 25% chance each cell is filled
    for x, y, z in grid:
        if y == 0:  # Fill the bottom plane completely
            grid[x, y, z] = 1
        else:
            grid[x, y, z] = 1 if ti.random(ti.f32) < 0.05 else 0

@ti.kernel
def get_cube_centers():
    idx = 0
    for x, y, z in ti.ndrange(grid_size, grid_size, grid_size):
        if grid[x, y, z] == 1:  # If the cell is filled
            # Map grid coordinates to world space (5x5x5 space)
            center = ti.Vector([
                (x + 0.5) * 5.0 / grid_size - 2.5,
                (y + 0.5) * 5.0 / grid_size - 2.5,
                (z + 0.5) * 5.0 / grid_size - 2.5
            ])
            cube_centers[idx] = center
            idx += 1

@ti.kernel
def get_cube_materials():
    idx = 0
    for x, y, z in ti.ndrange(grid_size, grid_size, grid_size):
        if grid[x, y, z] == 1:  # If the cell is filled
            if y == 0:  # If it's the bottom plane
                cube_materials[idx] = 2  # Green material
            else:
                # Assign a random material type (1 to 5)
                cube_materials[idx] = int(ti.random(ti.f32) * 5) + 1  # Random number between 1 and 5
            idx += 1

initialize_grid()
get_cube_centers()
get_cube_materials()

@ti.func
def shade(material: ti.i32, normal: ti.types.vector(3, ti.f32), camera: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    color_vector = ti.Vector([0.0, 0.0, 0.0])  # Default color vector
    if material == 1:
        color_vector = ti.Vector([1.0, 0.0, 0.0])  # Red for material 1
    elif material == 2:
        color_vector = ti.Vector([0.0, 1.0, 0.0])  # Green for material 2
    elif material == 3:
        color_vector = ti.Vector([0.0, 0.0, 1.0])  # Blue for material 3
    elif material == 4:
        color_vector = ti.Vector([1.0, 1.0, 0.0])  # Yellow for material 4
    elif material == 5:
        color_vector = ti.Vector([1.0, 0.0, 1.0])  # Magenta for material 5
    # Ambient, diffuse, and specular coefficients
    ambient = 0.3
    diffuse = 0.6
    specular = 0.9
    shininess = 32.0

    # Normalize the light direction and normal
    light_dir = light_dir.normalized()
    normal = normal.normalized()

    # Ambient component
    color = ambient * color_vector

    # Diffuse component
    diff = max(0.0, normal.dot(light_dir))
    color += diffuse * diff * color_vector

    # Specular component
    view_dir = camera # Assume the camera is at (0, 0, -1)
    reflect_dir = 2.0 * normal.dot(light_dir) * normal - light_dir
    spec = max(0.0, view_dir.dot(reflect_dir.normalized())) ** shininess
    color += specular * spec * ti.Vector([1.0, 1.0, 1.0])  # White specular highlight

    return color

@ti.kernel
def render_with_camera(camera: ti.types.vector(3, ti.f32)):
    for i, j in ti.ndrange(800, 800):  # Screen resolution
        ray = Ray(
            origin=camera,
            direction=ti.Vector([i / 400.0 - 1.0, j / 400.0 - 1.0, 1.0]).normalized()
        )
        color = ti.Vector([0.0, 0.0, 0.0])  # Default background color
        min_dist = float('inf')
        for idx in range(grid_size**3):
            cube_center = cube_centers[idx]
            if cube_center.norm() > 0:  # Check if the cube center is valid (non-zero)
                cube_size = 0.26
                cube_material = 1  # Assign a default material for now
                hit = intersect_cube(ray, cube_center, cube_size, cube_material)
                
                if hit.distance > 0.0 and hit.distance < min_dist:
                    min_dist = hit.distance
                    color = shade(cube_materials[idx], hit.normal, camera) 
                            
        # Write color to screen buffer
        screen_buffer[i, j] = color

@ti.func
def intersect_cube(ray: Ray, center: ti.types.vector(3, ti.f32), size: ti.f32, cube_material: ti.f32) -> Hit:
    t_min = -float('inf')
    t_max = float('inf')
    
    hit_occurred = False
    hit_point = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 0.0, 0.0])
    hit_distance = -1.0
    hit_material = -1

    for i in ti.static(range(3)):  # Check each axis
        inv_dir = 1.0 / ray.direction[i]
        t1 = (center[i] - size - ray.origin[i]) * inv_dir
        t2 = (center[i] + size - ray.origin[i]) * inv_dir
        
        if t1 > t2:
            t1, t2 = t2, t1
        
        if t1 > t_min:
            t_min = t1
            hit_normal = ti.Vector([0.0, 0.0, 0.0])
            hit_normal[i] = -1.0 if inv_dir < 0 else 1.0
        
        t_max = min(t_max, t2)
        if t_min > t_max:
            hit_occurred = False
        else:
            hit_occurred = True

    if hit_occurred:
        hit_point = ray.origin + t_min * ray.direction
        hit_distance = t_min
        hit_material = cube_material

    return Hit(hit_point, hit_normal, hit_distance, hit_material)

# Screen buffer to store the rendered image
screen_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(800, 800))

# Initialize GUI window
gui = ti.GUI("Ray Tracing a Cube", res=(800, 800))

# Camera position
camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
camera_pos[None] = ti.Vector([0.0, 0.0, -10.0])

# Main rendering loop
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        # Handle keyboard input for camera movement
        if e.key == ti.GUI.ESCAPE:
            gui.running = False  # Exit the program
        elif e.key == 'w':
            camera_pos[None][2] += 0.1  # Move camera forward
        elif e.key == 's':
            camera_pos[None][2] -= 0.1  # Move camera backward
        elif e.key == 'a':
            camera_pos[None][0] -= 0.1  # Move camera left
        elif e.key == 'd':
            camera_pos[None][0] += 0.1  # Move camera right
        elif e.key == ti.GUI.SPACE:
            camera_pos[None][1] += 0.1  # Move camera upward

    # Ensure the camera always looks at the center of the scene
    render_with_camera(camera_pos[None])  # Render the scene
    gui.set_image(screen_buffer.to_numpy())  # Display the rendered image
    gui.show()  # Update the GUI
