from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=1) # Create a scene, specifying the voxel edge and exposure.
scene.set_floor(0, (1.0, 1.0, 1.0)) # Height of the floor
scene.set_background_color((0.5, 0.5, 0.4)) # Color of the sky
scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6)) # Direction and color of the light

color_dirt = vec3(0.55, 0.27, 0.07)  # A brownish color for dirt
color_grass = vec3(0.13, 0.55, 0.13)  # A greenish color for grass
color_water = vec3(0.0, 0.5, 0.8)  # A bluish color for water
color_white = vec3(1.0, 1.0, 1.0)  # A pure white color

# Create a 3D list with dimensions 40x40x40
voxel_grid = ti.Vector.field(3, dtype=ti.i32, shape=(40, 20, 40))

@ti.func
def initgrid():
    # Map each cell of the grid to a 3x3x3 space in the range -60 to 60
    for x, y, z in voxel_grid:
        world_x = x * 3 - 60
        world_y = y * 3
        world_z = z * 3 - 60
        voxel_grid[x, y, z] = ivec3(world_x, world_y, world_z)  # Initialize with a default value

@ti.func
def fill_cell(x, y, z, type):
    if type == 1:
        make_grass_block(x, y, z)
    elif type == 2:
        make_river_block(x, y, z)
    else:
        create_block(voxel_grid[x, y, z], ivec3(3, 3, 3), color_white, vec3(0.0), 0)

@ti.func
def make_grass_block(x, y, z):
    create_block(voxel_grid[x, y, z], ivec3(3, 3, 3), color_grass, vec3(0.1))  # Top layer of grass
    create_block(voxel_grid[x, y, z], ivec3(3, 2, 3), color_dirt, vec3(0.2))  # Top layer of grass

@ti.func
def make_river_block(x, y, z):
    create_block(voxel_grid[x, y, z], ivec3(3, 2, 3), color_water, vec3(0.1))  # Top layer of grass



@ti.func 
def create_block(pos, size, color, color_noise, brightness=1):
    for I in ti.grouped(
        ti.ndrange((pos[0], pos[0] + size[0]),
                   (pos[1], pos[1] + size[1]),
                   (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, brightness, color + color_noise * ti.random())

@ti.kernel
def initialize_voxels():
    initgrid()
    for x, z in ti.ndrange(voxel_grid.shape[0], voxel_grid.shape[2]):
        fill_cell(x, 0, z, 1 if (x + z) < 50 else 2)



initialize_voxels()

scene.finish()