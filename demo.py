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

@ti.kernel
def build_vertical_pipe(scene: ti.template(), x: int, y: int, z: int):
    # connects top & bottom
    scene.set_voxel(ivec3(x,y,z)+ ivec3(1, 0, 1), 1, green)
    scene.set_voxel(ivec3(x,y,z)+ ivec3(1, 1, 1), 1, green)
    scene.set_voxel(ivec3(x,y,z) + ivec3(1, 2, 1), 1, green)

@ti.kernel
def build_horizontal_pipe_ew(scene: ti.template(), x: int, y: int, z: int):
    # connects east & west
    scene.set_voxel(ivec3(x, y, z) + ivec3(0, 1, 1), 1, blue)
    scene.set_voxel(ivec3(x, y, z) + ivec3(1, 1, 1), 1, blue)
    scene.set_voxel(ivec3(x, y, z) + ivec3(2, 1, 1), 1, blue)

@ti.kernel
def build_horizontal_pipe_ns(scene: ti.template(), x: int, y: int, z: int):
    # connects north & south
    scene.set_voxel(ivec3(x, y, z) + ivec3(1, 1, 0), 1, red)
    scene.set_voxel(ivec3(x, y, z) + ivec3(1, 1, 1), 1, red)
    scene.set_voxel(ivec3(x, y, z) + ivec3(1, 1, 2), 1, red)

@ti.kernel
def build_T_pipe_tbe(scene: ti.template(), x: int, y: int, z: int):
    # connects top & bottom & east
    scene.set_voxel(ivec3(x,y,z)+ ivec3(1, 0, 1), 1, yellow)
    scene.set_voxel(ivec3(x,y,z)+ ivec3(1, 1, 1), 1, yellow)
    scene.set_voxel(ivec3(x,y,z) + ivec3(1, 2, 1), 1, yellow)
    scene.set_voxel(ivec3(x,y,z)+ ivec3(2, 1, 1), 1, yellow)

@ti.kernel
def build_T_pipe_ewt(scene: ti.template(), x: int, y: int, z: int):
    # connects east & west & top
    scene.set_voxel(ivec3(x, y, z) + ivec3(0, 1, 1), 1, purple)
    scene.set_voxel(ivec3(x, y, z) + ivec3(1, 1, 1), 1, purple)
    scene.set_voxel(ivec3(x, y, z) + ivec3(2, 1, 1), 1, purple)
    scene.set_voxel(ivec3(x, y, z) + ivec3(1, 2, 1), 1, purple)

@ti.kernel
def build_T_pipe_nsb(scene: ti.template(), x: int, y: int, z: int):
    # connects north & south & bottom
    scene.set_voxel(ivec3(x, y, z) + ivec3(1, 1, 0), 1, cyan)
    scene.set_voxel(ivec3(x, y, z) + ivec3(1, 1, 1), 1, cyan)
    scene.set_voxel(ivec3(x, y, z) + ivec3(1, 1, 2), 1, cyan)
    scene.set_voxel(ivec3(x, y, z) + ivec3(1, 0, 1), 1, cyan)

@ti.kernel
def build_crosssection_pipe(scene: ti.template(), x: int, y: int, z: int):
    # connects top, bottom, east, west, north, south
    scene.set_voxel(ivec3(x,y,z)+ ivec3(1, 0, 1), 1, orange)
    scene.set_voxel(ivec3(x,y,z)+ ivec3(1, 1, 1), 1, orange)
    scene.set_voxel(ivec3(x,y,z) + ivec3(1, 2, 1), 1, orange)
    scene.set_voxel(ivec3(x,y,z)+ ivec3(2, 1, 1), 1, orange)
    scene.set_voxel(ivec3(x,y,z)+ ivec3(1, 1, 2), 1, orange)
    scene.set_voxel(ivec3(x,y,z)+ ivec3(0, 1, 1), 1, orange)
    scene.set_voxel(ivec3(x,y,z)+ ivec3(1, 1, 0), 1, orange)

@ti.kernel
def build_empty(scene: ti.template(), x: int, y: int, z: int):
    for i, j, k in ti.ndrange(2, 2, 2):
        scene.set_voxel(ivec3(x,y,z)+vec3(i,j,k), 0, vec3(0.0, 0.0, 0.0))

# @ti.kernel
# def build_vertical_stopper_t(scene: ti.template(), x: int, y: int, z: int):
#     ## connects top 
#     scene.set_voxel(ivec3(x,y,z) + ivec3(1, 2, 1), 1, dark_green)

# @ti.kernel
# def build_vertical_stopper_b(scene: ti.template(), x: int, y: int, z: int):
#     ## connects bottom 
#     scene.set_voxel(ivec3(x,y,z) + ivec3(1, 0, 1), 1, dark_green)

# @ti.kernel
# def build_horizontal_stopper_e(scene: ti.template(), x: int, y: int, z: int):
#     ## connects east
#     scene.set_voxel(ivec3(x,y,z) + ivec3(0, 0, 1), 1, dark_green)

# @ti.kernel
# def build_horizontal_stopper_w(scene: ti.template(), x: int, y: int, z: int):
#     ## connects west 
#     scene.set_voxel(ivec3(x,y,z) + ivec3(2, 0, 1), 1, dark_green)

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



# Example usage for path blocks:
possible_blocks = [
"vertical_pipe", #connects top & bottom
 "horizontal_pipe_ew", #connects east & west
 "horizontal_pipe_ns", #connects north & south
 "T_pipe_tbe", #connects top & bottom & east
 "T_pipe_ewt", #connects east & west & top
 "T_pipe_nsb", #connects north & south & bottom
# "crosssection_pipe", #connects top, bottom, east, west, north, south
#  "vertical_stopper_t", #connects top
#  "vertical_stopper_b", #connects bottom      
#  "horizontal_stopper_e", #connects east
#  "horizontal_stopper_w",  #connects west
 "empty" #connects west nowhere
 ]

pipe_connections = {
    "vertical_pipe": ["t", "b"],  # connects top & bottom
    "horizontal_pipe_ew": ["e", "w"],  # connects east & west
    "horizontal_pipe_ns": ["n", "s"],  # connects north & south
    "T_pipe_tbe": ["t", "b", "e"],  # connects top & bottom & east
    "T_pipe_ewt": ["e", "w", "t"],  # connects east & west & top
    "T_pipe_nsb": ["n", "s", "b"],  # connects north & south & bottom
    "crosssection_pipe": ["t", "b", "e", "w", "n", "s"],  # connects top, bottom, east, west, north, south
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
        (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add all pipe_connections that DON'T connect [e]
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
    # 'vertical_stopper_t': {
    #     (0,1,0): list(connections_with_b.keys()),  # Top neighbors - add pipe_connections that connect [b]
    #     (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors - add pipe_connections that DON'T connect [t]
    #     (1,0,0): list(connections_without_w.keys()),  # East neighbors - add all pipe_connections that DON'T connect [w]
    #     (0,0,1): list(connections_without_n.keys()),  # South neighbors - add all pipe_connections that DON'T connect [n]
    #     (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add all pipe_connections that DON'T connect [e]
    #     (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    # },
    # 'vertical_stopper_b': {
    #     (0,1,0): list(connections_without_b.keys()),  # Top neighbors - add pipe_connections that DON'T connect [b]
    #     (0,-1,0): list(connections_with_t.keys()),  # Bottom neighbors - add pipe_connections that connect [t]
    #     (1,0,0): list(connections_without_w.keys()),  # East neighbors - add all pipe_connections that DON'T connect [w]
    #     (0,0,1): list(connections_without_n.keys()),  # South neighbors - add all pipe_connections that DON'T connect [n]
    #     (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add all pipe_connections that DON'T connect [e]
    #     (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    # },
    # "horizontal_stopper_e": {
    #     (0,1,0): list(connections_without_b.keys()),  # Top neighbors - add pipe_connections that DON'T connect [b]
    #     (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors - add pipe_connections that DON'T connect [t]
    #     (1,0,0): list(connections_with_w.keys()),  # East neighbors - add all pipe_connections that connect [w]
    #     (0,0,1): list(connections_without_n.keys()),  # South neighbors - add all pipe_connections that DON'T connect [n]
    #     (-1,0,0): list(connections_without_e.keys()),  # West neighbors - add all pipe_connections that DON'T connect [e]
    #     (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    # },
    # "horizontal_stopper_w": {
    #     (0,1,0): list(connections_without_b.keys()),  # Top neighbors - add pipe_connections that DON'T connect [b]
    #     (0,-1,0): list(connections_without_t.keys()),  # Bottom neighbors - add pipe_connections that DON'T connect [t]
    #     (1,0,0): list(connections_without_w.keys()),  # East neighbors - add all pipe_connections that DON'T connect [w]
    #     (0,0,1): list(connections_without_n.keys()),  # South neighbors - add all pipe_connections that DON'T connect [n]
    #     (-1,0,0): list(connections_with_e.keys()),  # West neighbors - add all pipe_connections that connect [e]
    #     (0,0,-1): list(connections_without_s.keys())  # North neighbors - add all pipe_connections that DON'T connect [s]
    # },
}

allowed_neighbors = infer_allowed_neighbors(possible_blocks, allowed_partial)

block_types = [
    Block(name, globals()[f'build_{name}'], allowed_neighbors=allowed_neighbors[name])
    for name in possible_blocks
]

# --- Run WFC and build scene ---
wfc = WaveFunctionCollapse3D(10, 10, 10, block_types)  # 6x3x6 grid for demo
wfc.collapse()  # Seed for reproducibility
wfc.build_scene(scene)
scene.finish()

