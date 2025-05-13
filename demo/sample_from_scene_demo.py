import taichi as ti
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))  

from sample_block_extractor import SampleBlockExtractor
from wfc import WaveFunctionCollapse3D, build_kernel
from taichi.math import *
from scene import Scene


# Define unique colors for each block
green = vec3(0.0, 1.0, 0.0)
dark_green = vec3(0.0, 0.3, 0.0)
blue = vec3(0.0, 0.0, 1.0)
red = vec3(1.0, 0.0, 0.0)
yellow = vec3(1.0, 1.0, 0.0)
purple = vec3(0.5, 0.0, 0.5)
cyan = vec3(0.0, 1.0, 1.0)
orange = vec3(1.0, 0.5, 0.0)
dark_orange = vec3(0.3, 0.15, 0.0)
pink = vec3(1.0, 0.5, 1.0)
dark_grey = vec3(0.2, 0.2, 0.2)
light_blue = vec3(0.5, 0.5, 1.0)
dark_red = vec3(0.5, 0.0, 0.0)
light_green = vec3(0.5, 1.0, 0.5)
brown = vec3(0.4, 0.2, 0.1)
grey = vec3(0.5, 0.5, 0.5)
white = vec3(1.0, 1.0, 1.0)
black = vec3(0.0, 0.0, 0.0)
lime_green = vec3(0.5, 1.0, 0.5)

def make_sample_scene():
    # Create a 4x4x4 scene with 4 channels (r,g,b,mat)
    arr = np.zeros((4, 4, 4, 4), dtype=np.float32)
    # Floor: solid material (1), gray color, at y=0
    arr[:, 0, :, 0:3] = (0.5, 0.5, 0.5)
    arr[:, 0, :, 3] = 1
    # Vertical pipe: light material (2), yellow color, at x=2, from y=1 to y=3
    arr[2, 1:4, 2, 0:3] = (1.0, 1.0, 0.2)
    arr[2, 1:4, 2, 3] = 2
    # Horizontal pipe: light material (2), cyan color, at y=2, from x=0 to x=3
    arr[0:4, 2, 1, 0:3] = (0.2, 1.0, 1.0)
    arr[0:4, 2, 1, 3] = 2
    # Solid block in a corner: solid material (1), red color, at (3,3,3)
    arr[3, 3, 3, 0:3] = (1.0, 0.2, 0.2)
    arr[3, 3, 3, 3] = 1
    # Another solid block: solid material (1), blue color, at (0,3,1)
    arr[0, 3, 1, 0:3] = (0.2, 0.2, 1.0)
    arr[0, 3, 1, 3] = 1
    # The rest is air (material 0)
    return arr

##################################
# A pink, green and blue U shape #
##################################

def make_sample_scene_2():
    arr = np.zeros((5, 5, 5, 4), dtype=np.float32)
    arr[:, 0, :, 0:3] = green
    arr[:, 0, :, 3] = 2
    arr[0, :, :, 0:3] = purple
    arr[0, :, :, 3] = 2
    arr[4, :, :, 0:3] = blue
    arr[4, :, :, 3] = 2
    return arr


##################################
# A cyan, red and orange T shape #
##################################

def make_sample_scene_3():
    arr = np.zeros((5, 5, 5, 4), dtype=np.float32)
    arr[0:2, 0, :, 0:3] = orange
    arr[0:2, 0, :, 3] = 2
    arr[2, :, :, 0:3] = cyan
    arr[2, :, :, 3] = 2
    arr[3:5, 0, :, 0:3] = red
    arr[3:5, 0, :, 3] = 2
    return arr


####################################
# open white cube with black edges #
####################################

def make_sample_scene_4():
    arr = np.zeros((5, 5, 5, 4), dtype=np.float32)
    arr[:, :4, :, 0:3] = white
    arr[:, :4, :, 3] = 2
    arr[0:5, 0:5, 0, 0:3] = black
    arr[0:5, 0:5, 0, 3] = 1
    arr[0:5, 0, 0:5, 0:3] = black
    arr[0:5, 0, 0:5, 3] = 1
    arr[0, 0:5, 0:5, 0:3] = black
    arr[0, 0:5, 0:5, 3] = 1

    return arr


#call this function to visualize the blocks and their neighbors in the same scene
def block_debugger_and_viewer_in_scene(scene, sample_scene, block_objects, base_z=30, pad: int = 2):
    if callable(sample_scene):
        sample_scene = sample_scene()

    block_size = block_objects[0].data.shape
    build_kernel(scene, 0, 0, base_z + 20, sample_scene)

    name_to_block = {b.name: b for b in block_objects}

    offsets = {
        (1, 0, 0): (block_size[0] + pad, 0, 0),    # east
        (-1, 0, 0): (-(block_size[0] + pad), 0, 0), # west
        (0, 1, 0): (0, block_size[1] + pad, 0),     # up
        (0, -1, 0): (0, -(block_size[1] + pad), 0), # down
        (0, 0, 1): (0, 0, block_size[2] + pad),     # south
        (0, 0, -1): (0, 0, -(block_size[2] + pad)), # north
    }

    stride = (block_size[0] + pad) * 4

    for i, block in enumerate(block_objects):
        origin = ((i - len(block_objects) // 2) * stride, len(block.allowed_neighbors[(0, -1, 0)]) * offsets[(0, 1, 0)][1], base_z)
        block.build(scene, origin)

        for direction, neighbors in block.allowed_neighbors.items():
            off = offsets.get(direction)
            if off is None:
                continue

            for n_idx, neighbor_name in enumerate(neighbors):
                neighbor_block = name_to_block[neighbor_name]

                if direction in ((0, 1, 0), (0, -1, 0)):
                    stack = (0, (n_idx) * (block_size[2] + pad), 0 )
                else:
                    stack = (0, (n_idx) * (block_size[1] + pad), 0)

                pos = (
                    origin[0] + off[0] + stack[0],
                    origin[1] + off[1] + stack[1],
                    origin[2] + off[2] + stack[2],
                )

                neighbor_block.build(scene, pos)

if __name__ == '__main__':
    # Example usage:
    sample_scene = make_sample_scene()
    # sample_scene = np.load("example_castle_scene.npy")
    block_shape = (4, 2, 4)
    similarity_threshold = 0.90
    neighbor_distance = 1
    material_compatibility_map = {
        frozenset([0, 0]): 1.0,
        frozenset([1, 1]): 1.0,
        frozenset([2, 2]): 1.0,
        # frozenset([0, 1]): 0.5,
        # frozenset([0, 2]): 1.0,
        frozenset([1, 2]): 0.0,
    }
    seed = 42
    allow_repeated_blocks = False  # Set to True to allow repeated blocks, False for unique blocks only
    enforce_ground_constraint = True  # Set to True to enforce ground constraint

    extractor = SampleBlockExtractor(
        sample_scene,
        block_shape,
        similarity_threshold=similarity_threshold,
        neighbor_distance=neighbor_distance,
        material_compatibility_map=material_compatibility_map,
        allow_repeated_blocks=allow_repeated_blocks,
    )
    block_objects = extractor.get_block_objects()
    extractor.save_block_objects("blocks.json")
    print(f"Extracted {len(block_objects)} unique blocks.")
    wfc = WaveFunctionCollapse3D(4, 20, 4, block_objects, seed=seed, enforce_ground_constraint=enforce_ground_constraint)

    scene = Scene(voxel_edges=0.1, exposure=1)
    scene.set_floor(0, (1.0, 1.0, 1.0))
    scene.set_background_color((0.5, 0.5, 0.4))
    scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6))

    # Build block debugger visualization at z=30
    block_debugger_and_viewer_in_scene(
        scene,
        sample_scene,
        block_objects,
        base_z=40)

    wfc.collapse()
    wfc.build_scene(scene)
    scene.finish()