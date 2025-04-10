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
color_brick_red = vec3(0.8, 0.25, 0.33)  # A reddish color for bricks
color_straw = vec3(0.93, 0.87, 0.51)  # A straw-like yellowish color


@ti.func 
def create_block(pos, size, color, color_noise, brightness=1):
    for I in ti.grouped(
        ti.ndrange((pos[0], pos[0] + size[0]),
                   (pos[1], pos[1] + size[1]),
                   (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, brightness, color + color_noise * ti.random())

@ti.func
def create_dirtblock(pos):
    # Create the dirt layers
    create_block(pos, (4, 3, 4), color_dirt, 0.05)
    # Create the grass layer on top
    create_block(pos + ivec3(0, 3, 0), (4, 1, 4), color_grass, 0.02)

@ti.func
def create_riverblock(pos):
    create_block(pos, (4, 3, 4), color_water, 0.03)

@ti.func
def create_hutblock(pos):
    # Create the base of the hut
    create_block(pos, (8, 5, 8), color_brick_red, 0.03)
    # Create bright yellow squares in the middle of each face of the hut
    bright_yellow = vec3(1.0, 1.0, 0.0)
    # Front face
    create_block(pos + ivec3(2, 2, 0), (3, 1, 1), bright_yellow, 0.0, 2)
    # Back face
    create_block(pos + ivec3(2, 2, 7), (3, 1, 1), bright_yellow, 0.0, 2)
    # Left face
    create_block(pos + ivec3(0, 2, 2), (1, 1, 1), bright_yellow, 0.0, 2)
    # Right face
    create_block(pos + ivec3(7, 2, 2), (1, 1, 1), bright_yellow, 0.0, 2)
    # Top face
    create_block(pos + ivec3(2, 4, 2), (3, 1, 1), bright_yellow, 0.0, 2)
    
    # Create the triangular roof
    for y in range(5):
        size = 8 - 2 * y  # Roof gets smaller as it goes up
        create_block(pos + ivec3(y, 5 + y, y), (size, 1, size), color_straw, 0.02)


@ti.kernel
def initialize_voxels():
    for x, z in ti.ndrange((-60, 60), (-60, 60)):
        if x % 4 == 0 and z % 4 == 0:
            if (x**2 + z**2) > 30**2:  # Leave a small circle in the middle blank
                create_dirtblock(ivec3(x, 0, z))
            else:
                create_riverblock(ivec3(x, 0, z))
    create_hutblock(ivec3(-50, 4, 0))


    
    




initialize_voxels()

scene.finish()