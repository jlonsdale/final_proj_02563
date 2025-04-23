import taichi as ti
from taichi.math import vec3
from wfc import Block, WaveFunctionCollapse3D

# --- Block recipes for demo (adapted from example_city.py) ---
@ti.kernel
def build_grass_block(scene: ti.template(), x: int, y: int, z: int, color: ti.template()):
    for dx, dy, dz in ti.ndrange(3, 3, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color)

@ti.kernel
def build_dirt_block(scene: ti.template(), x: int, y: int, z: int, color: ti.template()):
    for dx, dz, dy in ti.ndrange(3, 3, 2):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color)

@ti.kernel
def build_water_block(scene: ti.template(), x: int, y: int, z: int, color: ti.template()):
    for dx, dz, dy in ti.ndrange(3, 3, 2):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color)
    for dx, dz in ti.ndrange(3, 3):
        scene.set_voxel((x*3+dx, y*3+2, z*3+dz), 1, vec3(1.0, 1.0, 1.0))

@ti.kernel
def build_path_block(scene: ti.template(), x: int, y: int, z: int, color: ti.template()):
    for dx, dz, dy in ti.ndrange(3, 3, 2):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, vec3(0.45, 0.17, 0.05))
    for dx in range(3):
        scene.set_voxel((x*3+dx, y*3+2, z*3+1), 1, color)

# --- Block color definitions ---
color_dirt = vec3(0.45, 0.17, 0.05)
color_grass = vec3(0.03, 0.45, 0.03)
color_water = vec3(0.0, 0.5, 0.8)
color_path = vec3(0.8, 0.6, 0.2)

# --- Scene setup ---
ti.init(arch=ti.gpu)
from scene import Scene
scene = Scene(voxel_edges=0, exposure=1)
scene.set_floor(0, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6))

# --- Block types ---
block_types = [
    Block('grass', color_grass, build_grass_block),
    Block('dirt', color_dirt, build_dirt_block),
    Block('water', color_water, build_water_block),
    Block('path', color_path, build_path_block),
]

# --- Run WFC and build scene ---
wfc = WaveFunctionCollapse3D(10, 10, block_types)
wfc.collapse()
wfc.build_scene(scene)
scene.finish()
