import numpy as np
import taichi as ti
from taichi.math import vec3
from wfc import Block, WaveFunctionCollapse3D
from taichi.math import *
from scene import Scene

ti.init(arch=ti.gpu)  # Initialize Taichi

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


# --- Scene setup ---
scene = Scene(voxel_edges=0, exposure=1)
scene.set_floor(0, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6))

# --- Block types ---

def infer_allowed_neighbors(blocks, allowed_partial, directions=[(1,0,0),(0,1,0)]):
    """
    Given a dict of allowed_neighbors for (1,0,0) and (0,1,0), infer (-1,0,0) and (0,-1,0).
    Returns a new dict with all four directions (in 3D, dz=0).
    """
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
    "bend_pipe_tn": ["t", "n"],  # connects top & north

    # "vertical_stopper_t": ["t"],  # connects top
    # "vertical_stopper_b": ["b"],  # connects bottom
    # "horizontal_stopper_e": ["e"],  # connects east
    # "horizontal_stopper_w": ["w"],  # connects west
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
        (0,1,0): list(connections_with_b.keys()),  # Top neighbors - add pipe_connections that connect [b]
        (0,-1,0): list(connections_with_t.keys()),  # Bottom neighbors - add pipe_connections that connect [t]
        (1,0,0): list(connections_without_w.keys()),  # East neighbors - add all pipe_connections that DON'T connect [w]
        (0,0,1): list(connections_without_n.keys()),  # South neighbors - add all pipe_connections that DON'T connect [n]
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add all pipe_connections that DON'T connect [e]
        (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    },
    'vertical_pipe_with_light': {
        (0,1,0): list(connections_with_b.keys()),  # Top neighbors - add pipe_connections that connect [b]
        (0,-1,0): list(connections_with_t.keys()),  # Bottom neighbors - add pipe_connections that connect [t]
        (1,0,0): list(connections_without_w.keys()),  # East neighbors - add all pipe_connections that DON'T connect [w]
        (0,0,1): list(connections_without_n.keys()),  # South neighbors - add all pipe_connections that DON'T connect [n]
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add all pipe_connections that DON'T connect [e]
        (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    },
    'horizontal_pipe_ew': {
        (0,1,0): list(connections_without_b.keys()),  # Top neighbors - add pipe_connections that DON'T connect [b]
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors - add pipe_connections that DON'T connect [t]
        (1,0,0): list(connections_with_w.keys()),  # East neighbors - add all pipe_connections that connect [w]
        (0,0,1): list(connections_without_n.keys()),  # South neighbors - add all pipe_connections that DON'T connect [n]
        (-1,0,0): list(connections_with_e.keys()),  # West neighbors - add all pipe_connections that connect [e]
        (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    },
    'horizontal_pipe_ns': {
        (0,1,0): list(connections_without_b.keys()),  # Top neighbors - add pipe_connections that DON'T connect [b]
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors - add pipe_connections that DON'T connect [t]
        (1,0,0): list(connections_without_w.keys()),  # East neighbors - add all pipe_connections that DON'T connect [w]
        (0,0,1): list(connections_with_n.keys()),  # South neighbors - add all pipe_connections that connect [n]
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add all pipe_connections that DON'T connect [e]
        (0,0,-1): list(connections_with_s.keys())  # North neighbors - add all pipe_connections that connect [s]
    },
    'T_pipe_tbe': {
        (0,1,0): list(connections_with_b.keys()),  # Top neighbors - add pipe_connections that connect [b]
        (0,-1,0): list(connections_with_t.keys()),  # Bottom neighbors - add pipe_connections that connect [t]
        (1,0,0): list(connections_with_w.keys()),  # East neighbors - add all pipe_connections that connect [w]
        (0,0,1): list(connections_without_n.keys()),  # South neighbors - add all pipe_connections that DON'T connect [n]
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add all pipe_connections that DON'T connect [e]
        (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    },
    'T_pipe_ewt': {
        (0,1,0): list(connections_with_b.keys()),  # Top neighbors - add pipe_connections that connect [b]
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors - add pipe_connections that DON'T connect [t]
        (1,0,0): list(connections_with_w.keys()),  # East neighbors - add all pipe_connections that connect [w]
        (0,0,1): list(connections_without_n.keys()),  # South neighbors - add all pipe_connections that DON'T connect [n]
        (-1,0,0): list(connections_with_e.keys()),  # West neighbors - add all pipe_connections that connect [e]
        (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    },
    "T_pipe_nsb": {
        (0,1,0): list(connections_without_b.keys()),  # Top neighbors - add pipe_connections that DON'T connect [b]
        (0,-1,0): list(connections_with_t.keys()),  # Bottom neighbors - add pipe_connections that connect [t]
        (1,0,0): list(connections_without_w.keys()),  # East neighbors - add all pipe_connections that DON'T connect [w]
        (0,0,1): list(connections_with_n.keys()),  # South neighbors - add all pipe_connections that connect [n]
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add pipe_connections that DON'T connect [e]
        (0,0,-1): list(connections_with_s.keys())  # North neighbors - add all pipe_connections that connect [s]
    },
    "crosssection_pipe": {
        (0,1,0): list(connections_with_b.keys()),  # Top neighbors - add pipe_connections that connect [b]
        (0,-1,0): list(connections_with_t.keys()),  # Bottom neighbors - add pipe_connections that connect [t]
        (1,0,0): list(connections_with_w.keys()),  # East neighbors - add all pipe_connections that connect [w]
        (0,0,1): list(connections_with_n.keys()),  # South neighbors - add all pipe_connections that connect [n]
        (-1,0,0): list(connections_with_e.keys()),  # West neighbors - add all pipe_connections that connect [e]
        (0,0,-1): list(connections_with_s.keys())  # North neighbors - add all pipe_connections that connect [s]
    },
    'empty': {
        (0,1,0): list(connections_without_b.keys()),  # Top neighbors - add pipe_connections that DON'T connect [b]
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors - add pipe_connections that DON'T connect [t]
        (1,0,0): list(connections_without_w.keys()),  # East neighbors - add all pipe_connections that DON'T connect [w]
        (0,0,1): list(connections_without_n.keys()),  # South neighbors - add all pipe_connections that DON'T connect [n]
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add all pipe_connections that DON'T connect [e]
        (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    },
    'bend_pipe_be': {
        (0,1,0): list(connections_without_b.keys()),  # Top neighbors - add pipe_connections that DON'T connect [b]
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors - add pipe_connections that DON'T connect [t]
        (1,0,0): list(connections_with_w.keys()),  # East neighbors - add all pipe_connections that connect [w]
        (0,0,1): list(connections_without_n.keys()),  # South neighbors - add all pipe_connections that DON'T connect [n]
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add all pipe_connections that DON'T connect [e]
        (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    },
    "bend_pipe_tn": {
        (0,1,0): list(connections_with_b.keys()),  # Top neighbors - add pipe_connections that connect [b]
        (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors - add pipe_connections that DON'T connect [t]
        (1,0,0): list(connections_without_w.keys()),  # East neighbors - add all pipe_connections that DON'T connect [w]
        (0,0,1): list(connections_with_n.keys()),  # South neighbors - add all pipe_connections that connect [n]
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add pipe_connections that DON'T connect [e]
        (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    },
}

allowed_neighbors = infer_allowed_neighbors(pipe_connections.keys(), allowed_partial)

def make_block_data(voxels):
    """
    voxels: list of (i, j, k, mat, (r, g, b))
    Returns a (3,3,3,4) numpy array.
    """
    arr = np.zeros((3, 3, 3, 4), dtype=np.float32)
    for i, j, k, mat, color in voxels:
        arr[i, j, k, 0:3] = color
        arr[i, j, k, 3] = mat
    return arr

block_data_dict = {
    'vertical_pipe': make_block_data([
        (1, 0, 1, 1, (0.0, 1.0, 0.0)),
        (1, 1, 1, 1, (0.0, 1.0, 0.0)),
        (1, 2, 1, 1, (0.0, 1.0, 0.0)),
    ]),
    'vertical_pipe_with_light': make_block_data([
        (1, 0, 1, 1, (0.2, 0.2, 0.2)),
        (1, 1, 1, 2, (1.0, 1.0, 0.0)),
        (1, 2, 1, 1, (0.2, 0.2, 0.2)),
    ]),
    'horizontal_pipe_ew': make_block_data([
        (0, 1, 1, 1, (0.0, 0.0, 1.0)),
        (1, 1, 1, 1, (0.0, 0.0, 1.0)),
        (2, 1, 1, 1, (0.0, 0.0, 1.0)),
    ]),
    'horizontal_pipe_ns': make_block_data([
        (1, 1, 0, 1, (1.0, 0.0, 0.0)),
        (1, 1, 1, 1, (1.0, 0.0, 0.0)),
        (1, 1, 2, 1, (1.0, 0.0, 0.0)),
    ]),
    'T_pipe_tbe': make_block_data([
        (1, 0, 1, 1, (1.0, 1.0, 0.0)),
        (1, 1, 1, 1, (1.0, 1.0, 0.0)),
        (1, 2, 1, 1, (1.0, 1.0, 0.0)),
        (2, 1, 1, 1, (1.0, 1.0, 0.0)),
    ]),
    'T_pipe_ewt': make_block_data([
        (0, 1, 1, 1, (0.5, 0.0, 0.5)),
        (1, 1, 1, 1, (0.5, 0.0, 0.5)),
        (2, 1, 1, 1, (0.5, 0.0, 0.5)),
        (1, 2, 1, 1, (0.5, 0.0, 0.5)),
    ]),
    'T_pipe_nsb': make_block_data([
        (1, 1, 0, 1, (0.0, 1.0, 1.0)),
        (1, 1, 1, 1, (0.0, 1.0, 1.0)),
        (1, 1, 2, 1, (0.0, 1.0, 1.0)),
        (1, 0, 1, 1, (0.0, 1.0, 1.0)),
    ]),
    'crosssection_pipe': make_block_data([
        (1, 0, 1, 1, (1.0, 0.5, 0.0)),
        (1, 1, 1, 1, (1.0, 0.5, 0.0)),
        (1, 2, 1, 1, (1.0, 0.5, 0.0)),
        (2, 1, 1, 1, (1.0, 0.5, 0.0)),
        (1, 1, 2, 1, (1.0, 0.5, 0.0)),
        (0, 1, 1, 1, (1.0, 0.5, 0.0)),
        (1, 1, 0, 1, (1.0, 0.5, 0.0)),
    ]),
    'bend_pipe_be': make_block_data([
        (1, 0, 1, 1, (0.3, 0.15, 0.0)),
        (1, 1, 1, 1, (0.3, 0.15, 0.0)),
        (2, 1, 1, 1, (0.3, 0.15, 0.0)),
    ]),
    'bend_pipe_tn': make_block_data([
        (1, 2, 1, 1, (1.0, 0.5, 1.0)),
        (1, 1, 1, 1, (1.0, 0.5, 1.0)),
        (1, 1, 0, 1, (1.0, 0.5, 1.0)),
    ]),
    'empty': make_block_data([]),
}

block_types = [
    Block(name, block_data_dict[name], allowed_neighbors=allowed_neighbors[name])
    for name in pipe_connections.keys()
]

# Display each block type in the scene for visualization
# for i, block in enumerate(block_types):
#     block.build(scene, (i * 4, 0, 0))  # Place each block 4 units apart along the x-axis

# --- Run WFC and build scene ---
wfc = WaveFunctionCollapse3D(10, 10, 10, block_types)  # 6x3x6 grid for demo
wfc.collapse()  # Seed for reproducibility
wfc.build_scene(scene)
scene.finish()

