from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=1) # Create a scene, specifying the voxel edge and exposure.
scene.set_floor(0, (1.0, 1.0, 1.0)) # Height of the floor
scene.set_background_color((0.5, 0.5, 0.4)) # Color of the sky
scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6)) # Direction and color of the light

color_dirt = vec3(0.45, 0.17, 0.05)  # A darker brownish color for dirt
color_grass = vec3(0.03, 0.45, 0.03)  # A darker greenish color for grass
color_water = vec3(0.0, 0.5, 0.8)  # A bluish color for water
color_white = vec3(1.0, 1.0, 1.0)  # A pure white color
color_path = vec3(0.6, 0.4, 0.2)  # A brownish color for the dirt path
color_trunk = vec3(0.3, 0.15, 0.05)  # A very dark brown color for the tree trunk
color_leaves = vec3(0.2, 0.8, 0.2)  # A light green color for the leaves

debug_position = ivec3(20, 0, 20)

voxel_grid = ti.Vector.field(3, dtype=ti.i32, shape=(40, 20, 40))

@ti.func
def initgrid():
    for x, y, z in voxel_grid:
        world_x = x * 3 - 60
        world_y = y * 3
        world_z = z * 3 - 60
        voxel_grid[x, y, z] = ivec3(world_x, world_y, world_z)  # Initialize with a default value

@ti.func
def fill_cell(x, y, z, type):
    if type == 1: # type 1 for grass
        make_grass_block(x, y, z)
    elif type == 2: # type 1 for river
        make_river_block(x, y, z)
    #add more types here
    else:
        create_block(voxel_grid[x, y, z], ivec3(3, 3, 3), color_white, vec3(0.0), 0)

@ti.func
def make_grass_block(x, y, z):
    create_block(voxel_grid[x, y, z], ivec3(3, 3, 3), color_grass, vec3(0.1)) 
    create_block(voxel_grid[x, y, z], ivec3(3, 2, 3), color_dirt, vec3(0.2))  

@ti.func
def make_river_block(x, y, z):
    create_block(voxel_grid[x, y, z], ivec3(3, 2, 3), color_water, vec3(0.1))  # Top layer of grass

@ti.func
def make_straight_path_block(x, y, z, direction):
    create_block(voxel_grid[x, y, z], ivec3(3, 3, 3), color_grass, vec3(0.1)) 
    if direction == 1:
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(0, 2, 1), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(2, 2, 1), 1, color_path)
    elif direction == 2:
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 0), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 2), 1, color_path)
    create_block(voxel_grid[x, y, z], ivec3(3, 2, 3), color_dirt, vec3(0.2))

@ti.func
def make_crosssection_path_block(x, y, z):
    # G P G  
    # P P P
    # G P G 

    # Center row and column form the path
    create_block(voxel_grid[x, y, z], ivec3(3, 3, 3), color_grass, vec3(0.1)) 
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(0, 2, 1), 1, color_path)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 0), 1, color_path)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, color_path)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 2), 1, color_path)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(2, 2, 1), 1, color_path)
    create_block(voxel_grid[x, y, z], ivec3(3, 2, 3), color_dirt, vec3(0.2))  

@ti.func
def make_corner_path_block(x, y, z, direction):
    create_block(voxel_grid[x, y, z], ivec3(3, 3, 3), color_grass, vec3(0.1)) 
    if direction == 1:
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 2), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(0, 2, 1), 1, color_path)
    elif direction == 2:
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 0), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(0, 2, 1), 1, color_path)
    elif direction == 3:
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 0), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(2, 2, 1), 1, color_path)
    elif direction == 4:
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 2), 1, color_path)
        scene.set_voxel(voxel_grid[x, y, z] + ivec3(2, 2, 1), 1, color_path)
    create_block(voxel_grid[x, y, z], ivec3(3, 2, 3), color_dirt, vec3(0.2))

@ti.func
def make_treetrunk_block(x, y, z):
    create_block(voxel_grid[x, y, z] + ivec3(1, 0, 1), ivec3(1, 3, 1), color_trunk, vec3(0.05))

@ti.func
def make_leaf_block(x, y, z):
    create_block(voxel_grid[x, y, z], ivec3(3, 3, 3), color_leaves, vec3(0.05))

@ti.func
def make_tree_top(x, y, z):
    create_block(voxel_grid[x, y, z], ivec3(3, 1, 3), color_leaves, vec3(0.05))
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 1, 1), 1, color_leaves)


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
    x, y, z = debug_position
    make_grass_block(x, y, z)  # Create a grass block
    make_river_block(x + 1, y, z)  # Create a river block
    make_straight_path_block(x + 2, y, z, 1)  # Create a straight path block with direction 1
    make_straight_path_block(x + 3, y, z, 2)  # Create a straight path block with direction 2
    make_crosssection_path_block(x + 4, y, z)  # Create a cross-section path block
    make_corner_path_block(x + 5, y, z, 1)  # Create a corner path block with direction 1
    make_corner_path_block(x + 6, y, z, 2)  # Create a corner path block with direction 2
    make_corner_path_block(x + 7, y, z, 3)  # Create a corner path block with direction 3
    make_corner_path_block(x + 8, y, z, 4)  # Create a corner path block with direction 4
    make_treetrunk_block(x,y+1,z)  # Create a tree trunk block
    make_leaf_block(x,y+2,z) 
    make_tree_top(x,y+3,z) 


initialize_voxels()

scene.finish()