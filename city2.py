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


step_array = ti.field(dtype=ti.i32, shape=67)

@ti.kernel
def initialize_step_array():
    for i in range(67):
        step_array[i] = i

initialize_step_array()


@ti.kernel
def initialize_voxels():
    for i in step_array:
        create_dirt_cube(ivec3(step_array[i], 1, step_array[i]))
        for j in range(step_array.shape[0]):
            create_dirt_cube(ivec3(step_array[i], 1, step_array[j]))

initialize_voxels()

scene.finish()