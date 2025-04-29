from scene import Scene
import taichi as ti
from taichi.math import *

ti.init(arch=ti.gpu)  # Initialize Taichi

scene = Scene(voxel_edges=0, exposure=1) # Create a scene, specifying the voxel edge and exposure.
scene.set_floor(0, (1.0, 1.0, 1.0)) # Height of the floor
scene.set_background_color((0.5, 0.5, 0.4)) # Color of the sky
scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6)) # Direction and color of the light

green = vec3(0.0, 1.0, 0.0)  # Define a green color
dark_green = vec3(0.0, 0.5, 0.0)  # Define a dark green color
voxel_grid = ti.Vector.field(3, dtype=ti.i32, shape=(40, 20, 40))


@ti.func
def initgrid():
    for I in ti.grouped(voxel_grid):
        voxel_grid[I] = ivec3(I.x * 3 - 60, I.y * 3, I.z * 3 - 60)  # Initialize with a default value

@ti.func
def build_vertical_pipe(x,y,z):   
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 0, 1), 1, green)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 1, 1), 1, green)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, green)

@ti.func
def build_horizontal_pipe(x,y,z):
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(0, 1, 1), 1, green)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 1, 1), 1, green)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(2, 1, 1), 1, green)

@ti.func
def build_T_pipe(x,y,z):
    build_vertical_pipe(x,y,z)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(2, 0, 1), 1, green)

@ti.func
def build_empty(x, y, z):
    for i, j, k in ti.ndrange(3, 3, 3):
        scene.set_voxel(voxel_grid[x, y, z]+vec3(i,j,k), 0, vec3(0.0, 0.0, 0.0))

@ti.func
def build_vertical_stopper_top(x, y, z):   
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 0, 1), 1, dark_green)

@ti.func
def build_vertical_stopper_bottom(x, y, z):   
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, dark_green)
    
block_types = {
    "vertical_pipe": build_vertical_pipe,
    "horizontal_pipe": build_horizontal_pipe,
    "T_pipe": build_T_pipe,
    "empty": build_empty,
    "vertical_stopper_top": build_vertical_stopper_top,
    "vertical_stopper_bottom": build_vertical_stopper_bottom
}

vertical_pipe_neighbours = {
    "top": ["T_pipe", "vertical_pipe, vertical_stopper_top"],
    "bottom": ["vertical_pipe, vertical_stopper_bottom"], 
    "north": ["empty"],
    "south": ["empty"],
    "east": ["empty"],
    "west": ["empty"]
}

horizontal_pipe_neighbours = {
    "top": ["empty"],
    "bottom": ["empty"],
    "north": ["empty"],
    "south": ["empty"],
    "east": ["horizontal_pipe"],
    "west": ["T_pipe", "horizontal_pipe"]
}

T_pipe_neighbours = {
    "top": ["T_pipe", "vertical_pipe, vertical_stopper_top"],
    "bottom": ["T_pipe", "vertical_pipe, vertical_stopper_bottom"],
    "north": ["empty"],
    "south": ["empty"],
    "east": ["horizontal_pipe"],
    "west": ["empty"]
}




@ti.kernel
def initialize_voxels():
    initgrid()
    build_vertical_pipe(1, 1, 1)
    build_vertical_stopper_top(1, 2, 1)
    build_vertical_stopper_bottom(1, 0, 1)


    
    
initialize_voxels()

scene.finish()