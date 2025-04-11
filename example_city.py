from scene import Scene
import taichi as ti
from taichi.math import *

ti.init(arch=ti.gpu)  # Initialize Taichi

scene = Scene(voxel_edges=0, exposure=1) # Create a scene, specifying the voxel edge and exposure.
scene.set_floor(0, (1.0, 1.0, 1.0)) # Height of the floor
scene.set_background_color((0.5, 0.5, 0.4)) # Color of the sky
scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6)) # Direction and color of the light

color_dirt = vec3(0.45, 0.17, 0.05)  # A darker brownish color for dirt
color_grass = vec3(0.03, 0.45, 0.03)  # A darker greenish color for grass
color_water = vec3(0.0, 0.5, 0.8)  # A bluish color for water
color_white = vec3(1.0, 1.0, 1.0)  # A pure white color
color_path = vec3(0.8, 0.6, 0.2)  # A yellowish-brown color for the dirt path
color_trunk = vec3(0.5, 0.3, 0.1)  # A lighter brown color for the tree trunk
color_leaves = vec3(0.2, 0.8, 0.2)  # A light green color for the leaves
color_white = vec3(1.0, 1.0, 1.0)  # A pure white color

debug_position = ivec3(20, 0, 20)

voxel_grid = ti.Vector.field(3, dtype=ti.i32, shape=(40, 20, 40))

@ti.func
def initgrid():
    for I in ti.grouped(voxel_grid):
        voxel_grid[I] = ivec3(I.x * 3 - 60, I.y * 3, I.z * 3 - 60)  # Initialize with a default value

@ti.func
def make_grass_block(x, y, z):
    create_block(voxel_grid[x, y, z], ivec3(3, 3, 3), color_grass, vec3(0.1)) 
    create_block(voxel_grid[x, y, z], ivec3(3, 2, 3), color_dirt, vec3(0.2))  

@ti.func
def make_river_block(x, y, z):
    create_block(voxel_grid[x, y, z], ivec3(3, 3, 3), color_white, vec3(0.0), 0) 
    create_block(voxel_grid[x, y, z], ivec3(3, 2, 3), color_water, vec3(0.0)) 
     
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
    create_block(voxel_grid[x, y, z] + ivec3(0, 2, 0), ivec3(3, 1, 3), color_grass, vec3(0.1)) 
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(0, 2, 1), 1, color_path)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 0), 1, color_path)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, color_path)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 2), 1, color_path)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(2, 2, 1), 1, color_path)
    create_block(voxel_grid[x, y, z], ivec3(3, 2, 3), color_dirt, vec3(0.2))  

@ti.func
def make_corner_path_block(x, y, z, direction):
    create_block(voxel_grid[x, y, z] + ivec3(0, 2, 0), ivec3(3, 1, 3), color_grass, vec3(0.1)) 
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
    create_block(voxel_grid[x, y, z] + ivec3(1, 0, 1), ivec3(1, 3, 1), color_trunk, vec3(0.0))

@ti.func
def make_leaf_block(x, y, z):
    create_block(voxel_grid[x, y, z], ivec3(3, 3, 3), color_leaves, vec3(0.0))

@ti.func
def make_tree_top(x, y, z):
    create_block(voxel_grid[x, y, z], ivec3(3, 1, 3), color_leaves, vec3(0.0))
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 1, 1), 1, color_leaves)


@ti.func 
def create_block(pos, size, color, color_noise, brightness=1):
    # disable random for now
    for I in ti.grouped(
        ti.ndrange((pos[0], pos[0] + size[0]),
                   (pos[1], pos[1] + size[1]),
                   (pos[2], pos[2] + size[2]))):
        noise = ti.select(color_noise != vec3(0.0), color_noise * ti.random(), vec3(0.0))
        scene.set_voxel(I, brightness, color+noise)  

@ti.kernel
def initialize_voxels():
    initgrid()
    for x, z in ti.ndrange((-10, 11), (-10, 11)):
        if z == -10 or z == 10 or x == -10 or x == 10:
            make_river_block(x + 21, 0, z + 21)
            make_river_block(x + 19, 0, z + 21)  
        make_grass_block(x + 20, 0, z + 20)
        if (x != 0 and z != 0) and ti.random() < 0.1:  
            tree_height = ti.random(ti.i32) % 3 + 2  
            for h in range(tree_height):
                make_treetrunk_block(x + 20, h, z + 20)
                make_tree_top(x + 20, tree_height, z + 20)
        
        if x == 0 and z == 0:
            make_crosssection_path_block(x + 20, 0, z + 20)
        elif x == 0 or z == 0:
            make_straight_path_block(x + 20, 0, z + 20, 1 if z == 0 else 2)








initialize_voxels()

scene.finish()