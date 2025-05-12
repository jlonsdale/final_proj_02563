import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))  # Dynamically add the utils directory to the path
print(sys.path)

from scene import Scene
import taichi as ti
from taichi.math import *
import os

ti.init(arch=ti.cpu)

GRID_W, GRID_D, GRID_H = 100, 100, 60

scene = Scene(exposure=1)
scene.set_floor(0, (0.8, 0.8, 0.8))
scene.set_background_color((0.6, 0.8, 1.0))
scene.set_directional_light((1, 1, 1), 0.3, (1, 1, 1))

# Materials
color_grass = vec3(0.3, 0.6, 0.3)
color_rock = vec3(0.4, 0.4, 0.4)
color_wall = vec3(1.0, 1.0, 1.0)
color_roof = vec3(0.2, 0.3, 0.6)
color_tower = vec3(0.9, 0.9, 0.95)
color_water = vec3(0.2, 0.4, 0.8)
color_red = vec3(1.0, 0.0, 0.0)
color_dark_green = vec3(0.0, 0.4, 0.1)

color_trunk = vec3(0.5, 0.3, 0.1)
color_leaves = vec3(0.2, 0.8, 0.2)
white = vec3(1.0, 1.0, 1.0)
black = vec3(0.1, 0.1, 0.1)

hill_centers = [
    ivec2(50, 40),  # center with castle
    ivec2(30, 20), ivec2(50, 20), ivec2(70, 20),
    ivec2(30, 40),               ivec2(70, 40),
    ivec2(30, 60), ivec2(50, 60), ivec2(70, 60),
]

@ti.func
def get_hill_height(x, z, cx, cz) -> int:
    fx = (x - cx) / 35.0
    fz = (z - cz) / 25.0
    r = fx * fx + fz * fz
    height = 0
    if r <= 1.3:
        height = int(18 + (1.3 - r) * 22)
    return height

@ti.func
def hill_height(x, z) -> int:
    fx = (x - 50) / 25.0   # X-center at 50
    fz = (z - 40) / 20.0   # Z-center shifted toward castle
    r = fx * fx + fz * fz
    height = 0
    if r <= 1.0:
        height = int(12 + (1.0 - r) * 25)
    return height

@ti.func
def create_block(pos: ivec3, size: ivec3, color, color_noise, brightness=1):
    for I in ti.grouped(ti.ndrange((pos[0], pos[0] + size[0]),
                                   (pos[1], pos[1] + size[1]),
                                   (pos[2], pos[2] + size[2]))):
        noise = ti.select(color_noise != vec3(0.0), color_noise * ti.random(), vec3(0.0))
        scene.set_voxel(I, brightness, color + noise)

@ti.func
def make_tree(center: ivec3):
    # Trunk (centered at base)
    create_block(center, ivec3(1, 4, 1), color_trunk, vec3(0.0))

    # Leaves block centered at trunk top (surrounding)
    create_block(center + ivec3(-1, 3, -1), ivec3(3, 3, 3), color_leaves, vec3(0.0))

    # Tree top crown
    create_block(center + ivec3(-1, 6, -1), ivec3(3, 1, 3), color_leaves, vec3(0.0))
    scene.set_voxel(center + ivec3(0, 7, 0), 1, color_leaves)


@ti.func
def create_tower(center, radius, height, roof_height):
    for y in range(height):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dz * dz <= radius * radius:
                    scene.set_voxel(center + ivec3(dx, y, dz), 1, color_tower)
    for i in range(roof_height):
        r = radius - i
        if r >= 0:
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx * dx + dz * dz <= r * r:
                        scene.set_voxel(center + ivec3(dx, height + i, dz), 1, color_roof)

@ti.func
def create_building(pos, width, depth, height, roof_height):
    for y in range(height):
        for dx in range(width):
            for dz in range(depth):
                scene.set_voxel(pos + ivec3(dx, y, dz), 1, color_wall)
    for i in range(roof_height):
        for dx in range(width - 2 * i):
            for dz in range(depth - 2 * i):
                scene.set_voxel(pos + ivec3(i + dx, height + i, i + dz), 1, color_roof)

@ti.func
def make_bush(center: ivec3):
    create_block(center, ivec3(2, 2, 2), color_dark_green, vec3(0.0))
    
@ti.func
def make_rock(center: ivec3):
    create_block(center, ivec3(2, 1, 2), color_rock, vec3(0.05))
    
@ti.func
def make_flower(center: ivec3):
    create_block(center, ivec3(1, 1, 1), color_red, vec3(0.05))
    
@ti.func
def make_tree_logs(center: ivec3):
    create_block(center, ivec3(1, 2, 1), color_trunk, vec3(0.05))

@ti.kernel
def initialize():
    # Build hill base
    for x, z in ti.ndrange(GRID_W, GRID_D):
        h = hill_height(x, z)
        for y in range(h):
            mat = color_rock if y < h - 3 else color_grass
            scene.set_voxel(ivec3(x, y, z), 1, mat)
            
            
    # Random tree scatter
    for i in range(150):  # increased from 100 to 500
        x = ti.random(ti.i32) % GRID_W
        z = ti.random(ti.i32) % GRID_D
        y = hill_height(x, z)

        if y > 10:
            # avoid castle zone
            if not (35 <= x <= 60 and 35 <= z <= 60):
                make_tree(ivec3(x, y, z))


    #make_tree(ivec3(30, 19, 35))
            
    #add rocks    
    for i in range(50):
        x = ti.random(ti.i32) % GRID_W
        z = ti.random(ti.i32) % GRID_D
        y = hill_height(x, z)
        if y > 10 and not (35 <= x <= 60 and 35 <= z <= 60):
            make_rock(ivec3(x, y, z))
            
    # Scatter bushes
    for i in range(80):
        x = ti.random(ti.i32) % GRID_W
        z = ti.random(ti.i32) % GRID_D
        y = hill_height(x, z)
        if y > 10 and not (35 <= x <= 60 and 35 <= z <= 60):
            make_bush(ivec3(x, y, z))
            
    #add flowers
    for i in range(50):
        x = ti.random(ti.i32) % GRID_W
        z = ti.random(ti.i32) % GRID_D
        y = hill_height(x, z)
        if y > 10 and not (35 <= x <= 60 and 35 <= z <= 60):
            make_flower(ivec3(x, y, z))
            
    #add logs
    for i in range(100):
        x = ti.random(ti.i32) % GRID_W
        z = ti.random(ti.i32) % GRID_D
        y = hill_height(x, z)
        if y > 10 and not (35 <= x <= 60 and 35 <= z <= 60):
            make_tree_logs(ivec3(x, y, z))
            
            
    base = ivec3(40, 30, 40)  # shifted to center better on hill

    # Central keep
    create_building(base, 10, 6, 12, 3)

    # Left and right wings (directly connected)
    #create_building(base + ivec3(-8, 0, 1), 8, 4, 10, 2)
    create_building(base + ivec3(10, 0, 1), 8, 4, 10, 2)

    # Rear hall connected to keep
    create_building(base + ivec3(2, 0, -6), 6, 4, 10, 2)
    
    create_building(base + ivec3(-2, 0, 1), 3, 4, 11, 10)

    # Towers merged with corners
    create_tower(base + ivec3(2, -2, -2), 2, 12, 4)  # Left tower touches left wing
    create_tower(base + ivec3(17, 0, 1), 2, 12, 4)  # Right tower touches right wing
    create_tower(base + ivec3(3, 0, -7), 2, 14, 5)  # Rear tower overlaps rear hall
    create_tower(base + ivec3(4, 0, 7), 2, 14, 7)  # Left tower touches left wing

initialize()
scene.finish()