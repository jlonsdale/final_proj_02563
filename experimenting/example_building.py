import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))  # Dynamically add the utils directory to the path
print(sys.path)

from scene import Scene
import taichi as ti
from taichi.math import *
import os

scene = Scene(voxel_edges=0, exposure=1)
scene.set_floor(0, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.7, 0.95))
scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6))

color_concrete = vec3(0.6, 0.6, 0.6)
color_roof = vec3(0.8, 0.3, 0.1)
color_water = vec3(0.2, 0.6, 0.9)
color_wood = vec3(0.3, 0.15, 0.05)
color_wall = vec3(1.0, 1.0, 1.0)

@ti.func
def create_crane_base(pos, height):
    for y in range(height):
        for dx in ti.static(range(-1, 2)):
            for dz in ti.static(range(-1, 2)):
                scene.set_voxel(pos + ivec3(dx, y, dz), 1, vec3(0.1, 0.05, 0.02))  # dark tower

@ti.func
def create_cylinder(pos, radius, height, color):
    for y in range(height):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dz * dz <= radius * radius:
                    scene.set_voxel(pos + ivec3(dx, y, dz), 1, color)

@ti.kernel
def initialize():
    # Water area (lower elevation: y = 0–1)
    for x, z in ti.ndrange((0, 40), (0, 20)):
        for y in range(2):
            scene.set_voxel(ivec3(x, y, z), 1, color_water)

    # Concrete base for buildings (from y=0 to y=3)
    for x, z in ti.ndrange((0, 40), (21, 40)):
        for y in range(0, 4):  # <— changed from (2, 4) to (0, 4)
            scene.set_voxel(ivec3(x, y, z), 1, color_concrete)

    # Retaining wall (still makes sense visually)
    for x in range(0, 40):
        for y in range(0, 4):  # also make wall reach full height
            scene.set_voxel(ivec3(x, y, 20), 1, color_concrete)

    # Buildings start above the platform (y=4+)
    center = ivec3(20, 4, 30)
    create_crane_base(center, 15)
    create_cylinder(center + ivec3(-5, 0, 0), 4, 10, vec3(0.6, 0.4, 0.3))
    create_cylinder(center + ivec3(5, 0, 0), 4, 10, vec3(0.6, 0.4, 0.3))

initialize()
scene.finish()