from scene import Scene
import taichi as ti
from taichi.math import *

ti.init(arch=ti.gpu)  # Initialize Taichi

scene = Scene(voxel_edges=0, exposure=1) # Create a scene, specifying the voxel edge and exposure.
scene.set_floor(0, (1.0, 1.0, 1.0)) # Height of the floor
scene.set_background_color((0.5, 0.5, 0.4)) # Color of the sky
scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6)) # Direction and color of the light




# Define the pattern
pattern = [
    "##########RRR####R###RRRR#####",
    "##########R##R###R###R##R#####",
    "##########R##R###R###R##R#####",
    "##########R##R###R###R##R#####",
    "##########RRR##RRRRR#R##R#####"
]



@ti.kernel
def initialize_voxels():
    # Loop through the pattern and set red voxels
    for y in ti.static(range(len(pattern))):
        for x in ti.static(range(len(pattern[y]))):
            if pattern[y][x] == 'R':
                scene.set_voxel(ivec3(x, y, 0), 2, vec3(0.8, 0.0, 0.0))  # Corporate red voxel





initialize_voxels()

scene.finish()