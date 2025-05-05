import taichi as ti
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))  # Dynamically add the utils directory to the path

from taichi.math import vec3
from wfc import Block, WaveFunctionCollapse3D
from taichi.math import *
from scene import Scene
from collections import Counter

ti.init(arch=ti.gpu)  # Initialize Taichi

block_data_dict = {
    "vertical_pipe": [[1, 0, 1], [1, 1, 1], [1, 2, 1]],  
    "horizontal_pipe_ew": [[0, 1, 1], [1, 1, 1], [2, 1, 1]],
    "horizontal_pipe_ns": [[1, 1, 0], [1, 1, 1], [1, 1, 2]],
    "T_pipe_tbe": [[1, 0, 1], [1, 1, 1], [1, 2, 1], [2, 1, 1]],
    "T_pipe_ewt": [[0, 1, 1], [1, 1, 1], [2, 1, 1], [1, 2, 1]],
    "T_pipe_nsb": [[1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 0, 1]],
    "crosssection_pipe": [[1, 0, 1], [1, 1, 1], [1, 2, 1], [2, 1, 1], [1, 1, 2], [0, 1, 1], [1, 1, 0]],
    "bend_pipe_be": [[1, 0, 1], [1, 1, 1], [2, 1, 1]],
    "bend_pipe_bw": [[1, 0, 1], [1, 1, 1], [0, 1, 1]],
    "bend_pipe_tn": [[1, 2, 1], [1, 1, 1], [1, 1, 0]],
    "bend_pipe_ts": [[1, 2, 1], [1, 1, 1], [1, 1, 2]],
    "vertical_stopper_t": [[1, 2, 1]],
    "vertical_stopper_b": [[1, 0, 1]],
    "horizontal_stopper_e": [[2, 0, 1]],
    "horizontal_stopper_w": [[0, 0, 1]],
    "empty": []
}

# --- Scene setup ---
scene = Scene(voxel_edges=0, exposure=1)
scene.set_floor(0, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6))

# --- Block types ---

def infer_allowed_neighbors(blocks, allowed_partial):
    # Start with a copy of the partial dict, using sets for easy updating
    full = {k: {d: set(v) for d, v in dct.items()} for k, dct in allowed_partial.items()}
    # Add missing directions
    for block in blocks:
        for d, axis in [((1,0,0), (-1,0,0)), ((0,0,1), (0,0,-1)), ((0,1,0), (0,-1,0))]:
            allowed = set()
            for other, other_dict in allowed_partial.items():
                if block in other_dict.get(d, []):
                    allowed.add(other)
            if axis in full[block]:
                full[block][axis].update(allowed)
            else:
                full[block][axis] = allowed
    # Convert sets back to lists
    return {k: {dir: list(v) for dir, v in d.items()} for k, d in full.items()}


pipe_connections = {
    "vertical_pipe": ["t", "b"],  # connects top & bottom
    "vertical_pipe_with_light": ["t", "b"],  # connects top & bottom
    "horizontal_pipe_ew": ["e", "w"],  # connects east & west
    "horizontal_pipe_ns": ["n", "s"],  # connects north & south
    "T_pipe_tbe": ["t", "b", "e"],  # connects top & bottom & east
    "T_pipe_ewt": ["e", "w", "t"],  # connects east & west & top
    "T_pipe_nsb": ["n", "s", "b"],  # connects north & south & bottom
    "crosssection_pipe": ["t", "b", "e", "w", "n", "s"],  # connects top, bottom, east, west, north, south
    "bend_pipe_be": ["b", "e"],  # connects bottom & east
    "bend_pipe_bw": ["b", "w"],  # connects bottom & west
    "bend_pipe_tn": ["t", "n"],  # connects top & north
    "bend_pipe_ts": ["t", "s"],  # connects top & south
    "vertical_stopper_t": ["t"],  # connects top
    "vertical_stopper_b": ["b"],  # connects bottom
    "horizontal_stopper_e": ["e"],  # connects east
    "horizontal_stopper_w": ["w"],  # connects west
    "empty": []  # connects nowhere

}

connections_without_n = {key: value for key, value in pipe_connections.items() if 'n' not in value}
connections_without_s = {key: value for key, value in pipe_connections.items() if 's' not in value}
connections_without_e = {key: value for key, value in pipe_connections.items() if 'e' not in value}
connections_without_w = {key: value for key, value in pipe_connections.items() if 'w' not in value}
connections_without_t = {key: value for key, value in pipe_connections.items() if 't' not in value}
connections_without_b = {key: value for key, value in pipe_connections.items() if 'b' not in value}

connections_with_n = {key: value for key, value in pipe_connections.items() if 'n' in value}
connections_with_s = {key: value for key, value in pipe_connections.items() if 's' in value}
connections_with_e = {key: value for key, value in pipe_connections.items() if 'e' in value}
connections_with_w = {key: value for key, value in pipe_connections.items() if 'w' in value}
connections_with_t = {key: value for key, value in pipe_connections.items() if 't' in value}
connections_with_b = {key: value for key, value in pipe_connections.items() if 'b' in value}

allowed_partial = {
    'vertical_pipe': {
        (0,1,0): list(connections_with_b.keys()),      # Top neighbors
        (0,-1,0): list(connections_with_t.keys()),     # Bottom neighbors
        (1,0,0): list(connections_without_w.keys()),   # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    'vertical_pipe_with_light': {
        (0,1,0): list(connections_with_b.keys()),      # Top neighbors
        (0,-1,0): list(connections_with_t.keys()),     # Bottom neighbors
        (1,0,0): list(connections_without_w.keys()),   # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    'horizontal_pipe_ew': {
        (0,1,0): list(connections_without_b.keys()),   # Top neighbors
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors
        (1,0,0): list(connections_with_w.keys()),      # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_with_e.keys()),     # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    'horizontal_pipe_ns': {
        (0,1,0): list(connections_without_b.keys()),   # Top neighbors
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors
        (1,0,0): list(connections_without_w.keys()),   # East neighbors
        (0,0,1): list(connections_with_n.keys()),      # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_with_s.keys())      # North neighbors
    },
    'T_pipe_tbe': {
        (0,1,0): list(connections_with_b.keys()),      # Top neighbors
        (0,-1,0): list(connections_with_t.keys()),     # Bottom neighbors
        (1,0,0): list(connections_with_w.keys()),      # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    'T_pipe_ewt': {
        (0,1,0): list(connections_with_b.keys()),      # Top neighbors
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors
        (1,0,0): list(connections_with_w.keys()),      # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_with_e.keys()),     # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    "T_pipe_nsb": {
        (0,1,0): list(connections_without_b.keys()),   # Top neighbors
        (0,-1,0): list(connections_with_t.keys()),     # Bottom neighbors
        (1,0,0): list(connections_without_w.keys()),   # East neighbors
        (0,0,1): list(connections_with_n.keys()),      # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_with_s.keys())      # North neighbors
    },
    "crosssection_pipe": {
        (0,1,0): list(connections_with_b.keys()),      # Top neighbors
        (0,-1,0): list(connections_with_t.keys()),     # Bottom neighbors
        (1,0,0): list(connections_with_w.keys()),      # East neighbors
        (0,0,1): list(connections_with_n.keys()),      # South neighbors
        (-1,0,0): list(connections_with_e.keys()),     # West neighbors
        (0,0,-1): list(connections_with_s.keys())      # North neighbors
    },
    'empty': {
        (0,1,0): list(connections_without_b.keys()),   # Top neighbors
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors
        (1,0,0): list(connections_without_w.keys()),   # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    'bend_pipe_be': {
        (0,1,0): list(connections_without_b.keys()),   # Top neighbors
        (0,-1,0): list(connections_with_t.keys()),     # Bottom neighbors
        (1,0,0): list(connections_with_w.keys()),      # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    "bend_pipe_bw": {
        (0,1,0): list(connections_without_b.keys()),   # Top neighbors
        (0,-1,0): list(connections_with_t.keys()),     # Bottom neighbors
        (1,0,0): list(connections_without_w.keys()),   # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_with_e.keys()),     # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    "bend_pipe_tn": {
        (0,1,0): list(connections_with_b.keys()),      # Top neighbors
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors
        (1,0,0): list(connections_without_w.keys()),   # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_with_s.keys())      # North neighbors
    },
    'bend_pipe_ts': {
        (0,1,0): list(connections_with_b.keys()),      # Top neighbors
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors
        (1,0,0): list(connections_without_w.keys()),   # East neighbors
        (0,0,1): list(connections_with_n.keys()),      # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    'vertical_stopper_t': {
        (0,1,0): list(connections_with_b.keys()),      # Top neighbors
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors
        (1,0,0): list(connections_without_w.keys()),   # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    'vertical_stopper_b': {
        (0,1,0): list(connections_without_b.keys()),   # Top neighbors
        (0,-1,0): list(connections_with_t.keys()),     # Bottom neighbors
        (1,0,0): list(connections_without_w.keys()),   # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    "horizontal_stopper_e": {
        (0,1,0): list(connections_without_b.keys()),   # Top neighbors
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors
        (1,0,0): list(connections_with_w.keys()),      # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
    "horizontal_stopper_w": {
        (0,1,0): list(connections_without_b.keys()),   # Top neighbors
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors
        (1,0,0): list(connections_without_w.keys()),   # East neighbors
        (0,0,1): list(connections_without_n.keys()),   # South neighbors
        (-1,0,0): list(connections_with_e.keys()),     # West neighbors
        (0,0,-1): list(connections_without_s.keys())   # North neighbors
    },
}

allowed_neighbors = infer_allowed_neighbors(pipe_connections.keys(), allowed_partial)

# Define block weights to control the frequency of certain blocks
block_weights = Counter({
    "vertical_pipe": 5,
    "horizontal_pipe_ew": 1,
    "horizontal_pipe_ns": 1,
    "T_pipe_tbe": 1,
    "T_pipe_ewt":1,
    "T_pipe_nsb": 1,
    "crosssection_pipe": 1,
    "bend_pipe_be": 5,
    "bend_pipe_bw": 5,
    "bend_pipe_tn": 5,
    "bend_pipe_ts": 5,
    "vertical_stopper_t": 0,
    "vertical_stopper_b": 0,
    "horizontal_stopper_e": 0,
    "horizontal_stopper_w": 0,
    "empty": 5
})

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

colors = [green, dark_green, blue, red, yellow, purple, cyan, orange, dark_orange, pink, dark_grey, light_blue, dark_red, light_green, brown, grey, white, black]

# Add a global flag to make all blocks green
make_all_green = True
material = 1  # Define a variable for material type

# Expand block types based on weights
expanded_block_types = [
    name for name, weight in block_weights.items() for _ in range(weight)
]

def make_block_data(voxels):
    arr = np.zeros((3, 3, 3, 4), dtype=np.float32)
    color = green if make_all_green else colors[np.random.randint(len(colors))]
    for i, j, k in voxels:
        arr[i, j, k, 0:3] = color
        arr[i, j, k, 3] = 1
    return arr

# Create block instances
block_types = []

for name in expanded_block_types:
    block_data = make_block_data(block_data_dict[name])
    block = Block(name, block_data, allowed_neighbors[name])
    block_types.append(block)
    
# --- Run WFC and build scene ---
wfc = WaveFunctionCollapse3D(10, 10, 2, block_types)  # 6x3x6 grid for demo
wfc.collapse()  # Seed for reproducibility
wfc.build_scene(scene)
scene.finish()

