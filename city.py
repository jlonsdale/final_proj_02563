from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=2) # Create a scene, specifying the voxel edge and exposure.
scene.set_floor(0, (1.0, 1.0, 1.0)) # Height of the floor
scene.set_background_color((0.5, 0.5, 0.4)) # Color of the sky
scene.set_directional_light((1, 1, -1), 0.2, (1, 0.8, 0.6)) # Direction and color of the light


color_dirt = vec3(0.55, 0.27, 0.07)  # A brownish color for dirt
color_grass = vec3(0.13, 0.55, 0.13)  # A greenish color for grass
color_water = vec3(0.0, 0.5, 0.8)  # A bluish color for water



@ti.func 
def create_block(pos, size, color, color_noise):
    for I in ti.grouped(
        ti.ndrange((pos[0], pos[0] + size[0]),
                   (pos[1], pos[1] + size[1]),
                   (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, 1, color + color_noise * ti.random())


@ti.func
def create_dirt_cube(base_pos):
    create_block(base_pos, (3, 2, 3), color_dirt, 0.3)
    create_block((base_pos[0], base_pos[1] + 2, base_pos[2]), (3, 1, 3), color_grass, 0.1)    

@ti.func
def create_river_cube(base_pos):
    create_block(base_pos, (3, 2, 3), color_water, 0.0)

@ti.func
def create_stone_wall(base_pos, facing_direction):
    for i in range(3):  # Height of the wall
        for j in range(1):  # Thickness of the wall
            if facing_direction == 1:
                create_block((base_pos[0] + j, base_pos[1] + i, base_pos[2]), (1, 1, 3), vec3(0.5, 0.5, 0.5), 0.1)
            elif facing_direction == 2:
                create_block((base_pos[0], base_pos[1] + i, base_pos[2] + j), (3, 1, 1), vec3(0.5, 0.5, 0.5), 0.1)


@ti.func
def create_bridge_cube(base_pos):
    create_block(base_pos, (3, 1, 3), vec3(0.4, 0.2, 0.1), vec3(0.2, 0.2, 0.0) * ti.random())

   

@ti.kernel
def initialize_voxels():
    for x in range(50): 
        for z in range(50):
            if 20 <= x <= 25:  # Create a wider river running through the middle
                create_river_cube((x, 0, z))
                if z == 43:  # Create a wider river running through the middle
                    create_bridge_cube((x, 3, z))  # Create a bridge above the river
            else:
                create_dirt_cube((x, 0, z))
    for x in range(20):
        for z in range(20):
            if x == 0 or x == 19 or z == 0 or z == 19:  # Create the square wall boundary
                create_stone_wall((x, 3, z), 1 if x == 0 or x == 19 else 2)
                create_stone_wall((x, 6, z), 1 if x == 0 or x == 19 else 2)  # Add another layer on top







initialize_voxels()

scene.finish()