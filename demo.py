import taichi as ti
from taichi.math import vec3
from wfc import Block, WaveFunctionCollapse3D

# --- Block recipes for demo (adapted from example_city.py) ---
@ti.kernel
def build_grass_block(scene: ti.template(), x: int, y: int, z: int):
    color_grass = vec3(0.03, 0.45, 0.03)
    color_dirt = vec3(0.45, 0.17, 0.05)
    for dx, dy, dz in ti.ndrange(3, 3, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_grass)
    for dx, dz, dy in ti.ndrange(3, 3, 2):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_dirt)

@ti.kernel
def build_dirt_block(scene: ti.template(), x: int, y: int, z: int):
    color_dirt = vec3(0.45, 0.17, 0.05)
    for dx, dy, dz in ti.ndrange(3, 2, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_dirt)

@ti.kernel
def build_water_block(scene: ti.template(), x: int, y: int, z: int):
    color_water = vec3(0.0, 0.5, 0.8)
    color_white = vec3(1.0, 1.0, 1.0)
    for dx, dy, dz in ti.ndrange(3, 2, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_water)
    for dx, dz in ti.ndrange(3, 3):
        scene.set_voxel((x*3+dx, y*3+2, z*3+dz), 0, color_white)

@ti.kernel
def build_path_block_x(scene: ti.template(), x: int, y: int, z: int):
    color_grass = vec3(0.03, 0.45, 0.03)
    color_dirt = vec3(0.45, 0.17, 0.05)
    color_path = vec3(0.8, 0.6, 0.2)
    # Grass base
    for dx, dy, dz in ti.ndrange(3, 3, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_grass)
    # Path in x direction (center row)
    for dx in range(3):
        scene.set_voxel((x*3+dx, y*3+2, z*3+1), 1, color_path)
    # Dirt under
    for dx, dy, dz in ti.ndrange(3, 2, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_dirt)

@ti.kernel
def build_path_block_z(scene: ti.template(), x: int, y: int, z: int):
    color_grass = vec3(0.03, 0.45, 0.03)
    color_dirt = vec3(0.45, 0.17, 0.05)
    color_path = vec3(0.8, 0.6, 0.2)
    # Grass base
    for dx, dy, dz in ti.ndrange(3, 3, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_grass)
    # Path in z direction (center column)
    for dz in range(3):
        scene.set_voxel((x*3+1, y*3+2, z*3+dz), 1, color_path)
    # Dirt under
    for dx, dy, dz in ti.ndrange(3, 2, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_dirt)

@ti.kernel
def build_path_block_cross(scene: ti.template(), x: int, y: int, z: int):
    color_grass = vec3(0.03, 0.45, 0.03)
    color_dirt = vec3(0.45, 0.17, 0.05)
    color_path = vec3(0.8, 0.6, 0.2)
    # Grass base
    for dx, dy, dz in ti.ndrange(3, 3, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_grass)
    # Cross path
    for dx in range(3):
        scene.set_voxel((x*3+dx, y*3+2, z*3+1), 1, color_path)
    for dz in range(3):
        scene.set_voxel((x*3+1, y*3+2, z*3+dz), 1, color_path)
    # Dirt under
    for dx, dy, dz in ti.ndrange(3, 2, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_dirt)

@ti.kernel
def build_path_block_corner1(scene: ti.template(), x: int, y: int, z: int):
    color_grass = vec3(0.03, 0.45, 0.03)
    color_dirt = vec3(0.45, 0.17, 0.05)
    color_path = vec3(0.8, 0.6, 0.2)
    # Grass base
    for dx, dy, dz in ti.ndrange(3, 3, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_grass)
    # Corner path (x+ and z+)
    scene.set_voxel((x*3+1, y*3+2, z*3+1), 1, color_path)
    scene.set_voxel((x*3+1, y*3+2, z*3+2), 1, color_path)
    scene.set_voxel((x*3+0, y*3+2, z*3+1), 1, color_path)
    # Dirt under
    for dx, dy, dz in ti.ndrange(3, 2, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_dirt)

@ti.kernel
def build_path_block_corner2(scene: ti.template(), x: int, y: int, z: int):
    color_grass = vec3(0.03, 0.45, 0.03)
    color_dirt = vec3(0.45, 0.17, 0.05)
    color_path = vec3(0.8, 0.6, 0.2)
    # Grass base
    for dx, dy, dz in ti.ndrange(3, 3, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_grass)
    # Corner path (x+ and z-)
    scene.set_voxel((x*3+1, y*3+2, z*3+0), 1, color_path)
    scene.set_voxel((x*3+1, y*3+2, z*3+1), 1, color_path)
    scene.set_voxel((x*3+0, y*3+2, z*3+1), 1, color_path)
    # Dirt under
    for dx, dy, dz in ti.ndrange(3, 2, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_dirt)

@ti.kernel
def build_path_block_corner3(scene: ti.template(), x: int, y: int, z: int):
    color_grass = vec3(0.03, 0.45, 0.03)
    color_dirt = vec3(0.45, 0.17, 0.05)
    color_path = vec3(0.8, 0.6, 0.2)
    # Grass base
    for dx, dy, dz in ti.ndrange(3, 3, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_grass)
    # Corner path (x- and z-)
    scene.set_voxel((x*3+1, y*3+2, z*3+0), 1, color_path)
    scene.set_voxel((x*3+1, y*3+2, z*3+1), 1, color_path)
    scene.set_voxel((x*3+2, y*3+2, z*3+1), 1, color_path)
    # Dirt under
    for dx, dy, dz in ti.ndrange(3, 2, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_dirt)

@ti.kernel
def build_path_block_corner4(scene: ti.template(), x: int, y: int, z: int):
    color_grass = vec3(0.03, 0.45, 0.03)
    color_dirt = vec3(0.45, 0.17, 0.05)
    color_path = vec3(0.8, 0.6, 0.2)
    # Grass base
    for dx, dy, dz in ti.ndrange(3, 3, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_grass)
    # Corner path (x- and z+)
    scene.set_voxel((x*3+1, y*3+2, z*3+1), 1, color_path)
    scene.set_voxel((x*3+1, y*3+2, z*3+2), 1, color_path)
    scene.set_voxel((x*3+2, y*3+2, z*3+1), 1, color_path)
    # Dirt under
    for dx, dy, dz in ti.ndrange(3, 2, 3):
        scene.set_voxel((x*3+dx, y*3+dy, z*3+dz), 1, color_dirt)

# --- Scene setup ---
ti.init(arch=ti.gpu)
from scene import Scene
scene = Scene(voxel_edges=0, exposure=1)
scene.set_floor(0, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6))

# --- Block types ---
grass_allowed = {
    (-1,0): ['grass','dirt'],
    (1,0): ['grass','dirt'],
    (0,-1): ['grass','dirt'],
    (0,1): ['grass','dirt']
}
dirt_allowed = {
    (-1,0): ['grass','dirt','water','path_x','path_z','path_cross','path_corner1','path_corner2','path_corner3','path_corner4'],
    (1,0):  ['grass','dirt','water','path_x','path_z','path_cross','path_corner1','path_corner2','path_corner3','path_corner4'],
    (0,-1): ['grass','dirt','water','path_x','path_z','path_cross','path_corner1','path_corner2','path_corner3','path_corner4'],
    (0,1):  ['grass','dirt','water','path_x','path_z','path_cross','path_corner1','path_corner2','path_corner3','path_corner4']
}
water_allowed = {
    (-1,0): ['water','dirt'],
    (1,0): ['water','dirt'],
    (0,-1): ['water','dirt'],
    (0,1): ['water','dirt']
}
path_x_allowed = {
    (-1,0): ['path_x','path_cross','path_corner1','path_corner2','path_corner3','path_corner4','dirt'],
    (1,0):  ['path_x','path_cross','path_corner1','path_corner2','path_corner3','path_corner4','dirt'],
    (0,-1): ['path_x','dirt','grass','path_corner1','path_corner2','path_corner3','path_corner4'],
    (0,1):  ['path_x','dirt','grass','path_corner1','path_corner2','path_corner3','path_corner4']
}
path_z_allowed = {
    (-1,0): ['path_z','dirt','grass','path_corner1','path_corner2','path_corner3','path_corner4'],
    (1,0):  ['path_z','dirt','grass','path_corner1','path_corner2','path_corner3','path_corner4'],
    (0,-1): ['path_z','path_cross','path_corner1','path_corner2','path_corner3','path_corner4','dirt'],
    (0,1):  ['path_z','path_cross','path_corner1','path_corner2','path_corner3','path_corner4','dirt']
}
path_cross_allowed = {
    (-1,0): ['path_x','path_cross','path_corner1','path_corner2','path_corner3','path_corner4'],
    (1,0):  ['path_x','path_cross','path_corner1','path_corner2','path_corner3','path_corner4'],
    (0,-1): ['path_z','path_cross','path_corner1','path_corner2','path_corner3','path_corner4'],
    (0,1):  ['path_z','path_cross','path_corner1','path_corner2','path_corner3','path_corner4']
}
path_corner1_allowed = {
    (-1,0): ['path_x','path_cross'],
    (1,0):  ['dirt','grass','path_z','path_corner2','path_corner3','path_corner4'],
    (0,-1): ['path_z','path_cross','path_corner2','path_corner3','path_corner4'],
    (0,1):  ['dirt','grass','path_x']
}
path_corner2_allowed = {
    (-1,0): ['path_x','path_cross','path_corner2','dirt','grass'],
    (1,0):  ['dirt','grass','path_z','path_cross','path_corner1','path_corner2','path_corner3','path_corner4'],
    (0,-1): ['path_z','path_cross','path_corner2','dirt','grass'],
    (0,1):  ['dirt','grass','path_x','path_cross','path_corner1','path_corner2','path_corner3','path_corner4']
}
path_corner3_allowed = {
    (-1,0): ['dirt','grass','path_x','path_cross','path_corner1','path_corner2','path_corner3','path_corner4'],
    (1,0):  ['path_x','path_cross','path_corner3','dirt','grass'],
    (0,-1): ['path_z','path_cross','path_corner3','dirt','grass'],
    (0,1):  ['dirt','grass','path_x','path_cross','path_corner1','path_corner2','path_corner3','path_corner4']
}
path_corner4_allowed = {
    (-1,0): ['dirt','grass','path_x','path_cross','path_corner1','path_corner2','path_corner3','path_corner4'],
    (1,0):  ['path_x','path_cross','path_corner4','dirt','grass'],
    (0,-1): ['dirt','grass','path_z','path_cross','path_corner1','path_corner2','path_corner3','path_corner4'],
    (0,1):  ['path_z','path_cross','path_corner4','dirt','grass']
}

block_types = [
    # Block('grass', build_grass_block, allowed_neighbors=grass_allowed),
    # Block('dirt', build_dirt_block, allowed_neighbors=dirt_allowed),
    # Block('water', build_water_block, allowed_neighbors=water_allowed),
    Block('path_x', build_path_block_x, allowed_neighbors=path_x_allowed),
    Block('path_z', build_path_block_z, allowed_neighbors=path_z_allowed),
    Block('path_cross', build_path_block_cross, allowed_neighbors=path_cross_allowed),
    # Block('path_corner1', build_path_block_corner1, allowed_neighbors=path_corner1_allowed),
    # Block('path_corner2', build_path_block_corner2, allowed_neighbors=path_corner2_allowed),
    # Block('path_corner3', build_path_block_corner3, allowed_neighbors=path_corner3_allowed),
    # Block('path_corner4', build_path_block_corner4, allowed_neighbors=path_corner4_allowed),
]

# --- Run WFC and build scene ---
wfc = WaveFunctionCollapse3D(10, 10, block_types)
wfc.collapse()
wfc.build_scene(scene)
scene.finish()
