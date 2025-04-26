import taichi as ti
from taichi.math import vec3
from wfc import Block, WaveFunctionCollapse3D
from taichi.math import *
from scene import Scene

ti.init(arch=ti.gpu)  # Initialize Taichi

green = vec3(0.0, 1.0, 0.0)  # Define a green color
dark_green = vec3(0.0, 0.5, 0.0)  # Define a dark green color
voxel_grid = ti.Vector.field(3, dtype=ti.i32, shape=(40, 20, 40))

@ti.func
def initgrid():
    for I in ti.grouped(voxel_grid):
        voxel_grid[I] = ivec3(I.x * 3 - 60, I.y * 3, I.z * 3 - 60)  # Initialize with a default value

@ti.kernel
def build_vertical_pipe(scene: ti.template(), x: int, y: int, z: int):   
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 0, 1), 1, green)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 1, 1), 1, green)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, green)

@ti.kernel
def build_horizontal_pipe(scene: ti.template(), x: int, y: int, z: int):
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(0, 1, 1), 1, green)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 1, 1), 1, green)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(2, 1, 1), 1, green)

@ti.kernel
def build_T_pipe(scene: ti.template(), x: int, y: int, z: int):
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 0, 1), 1, green)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 1, 1), 1, green)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, green)
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(2, 0, 1), 1, green)

@ti.kernel
def build_empty(scene: ti.template(), x: int, y: int, z: int):
    for i, j, k in ti.ndrange(3, 3, 3):
        scene.set_voxel(voxel_grid[x, y, z]+vec3(i,j,k), 0, vec3(0.0, 0.0, 0.0))

@ti.kernel
def build_vertical_stopper_top(scene: ti.template(), x: int, y: int, z: int): 
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 0, 1), 1, dark_green)

@ti.kernel
def build_vertical_stopper_bottom(scene: ti.template(), x: int, y: int, z: int):
    scene.set_voxel(voxel_grid[x, y, z] + ivec3(1, 2, 1), 1, dark_green)
    

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
        for d, axis in [((1,0,0), (-1,0,0)), ((0,0,1), (0,0,-1))]:
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
path_blocks = ["vertical_pipe", "horizontal_pipe", "T_pipe", "empty", "vertical_stopper_top", "vertical_stopper_bottom"]
allowed_partial = {
    'vertical_pipe': {
        (0,1,0): ['vertical_pipe', 'T_pipe', 'vertical_stopper_top'],  # Top neighbors
        (0,-1,0): ['vertical_pipe', 'T_pipe', 'vertical_stopper_bottom'],  # Bottom neighbors
        (1,0,0): ['horizontal_pipe', 'T_pipe'],  # East neighbors
        (0,0,1): ['horizontal_pipe', 'T_pipe'],  # South neighbors
        (-1,0,0): ['horizontal_pipe', 'T_pipe'],  # West neighbors
        (0,0,-1): ['horizontal_pipe', 'T_pipe']  # North neighbors
    },
    'horizontal_pipe': {
        (0,1,0): ['T_pipe', 'vertical_pipe'],  # Top neighbors
        (0,-1,0): ['T_pipe', 'vertical_pipe'],  # Bottom neighbors
        (1,0,0): ['horizontal_pipe', 'T_pipe'],  # East neighbors
        (0,0,1): ['horizontal_pipe', 'T_pipe'],  # South neighbors
        (-1,0,0): ['horizontal_pipe', 'T_pipe'],  # West neighbors
        (0,0,-1): ['horizontal_pipe', 'T_pipe']  # North neighbors
    },
    'T_pipe': {
        (0,1,0): ['vertical_pipe', 'T_pipe', 'vertical_stopper_top'],  # Top neighbors
        (0,-1,0): ['vertical_pipe', 'T_pipe', 'vertical_stopper_bottom'],  # Bottom neighbors
        (1,0,0): ['horizontal_pipe', 'T_pipe'],  # East neighbors
        (0,0,1): ['horizontal_pipe', 'T_pipe'],  # South neighbors
        (-1,0,0): ['horizontal_pipe', 'T_pipe'],  # West neighbors
        (0,0,-1): ['horizontal_pipe', 'T_pipe']  # North neighbors
    },
    'empty': {
        (0,1,0): ['empty'],  # Top neighbors
        (0,-1,0): ['empty'],  # Bottom neighbors
        (1,0,0): ['empty'],  # East neighbors
        (0,0,1): ['empty'],  # South neighbors
        (-1,0,0): ['empty'],  # West neighbors
        (0,0,-1): ['empty']  # North neighbors
    },
    'vertical_stopper_top': {
        (0,1,0): ['empty'],  # Top neighbors
        (0,-1,0): ['vertical_pipe', 'T_pipe'],  # Bottom neighbors
        (1,0,0): ['horizontal_pipe', 'T_pipe'],  # East neighbors
        (0,0,1): ['horizontal_pipe', 'T_pipe'],  # South neighbors
        (-1,0,0): ['horizontal_pipe', 'T_pipe'],  # West neighbors
        (0,0,-1): ['horizontal_pipe', 'T_pipe']  # North neighbors
    },
    'vertical_stopper_bottom': {
        (0,1,0): ['vertical_pipe', 'T_pipe'],  # Top neighbors
        (0,-1,0): ['empty'],  # Bottom neighbors
        (1,0,0): ['horizontal_pipe', 'T_pipe'],  # East neighbors
        (0,0,1): ['horizontal_pipe', 'T_pipe'],  # South neighbors
        (-1,0,0): ['horizontal_pipe', 'T_pipe'],  # West neighbors
        (0,0,-1): ['horizontal_pipe', 'T_pipe']  # North neighbors
    }
}

allowed_neighbors = infer_allowed_neighbors(path_blocks, allowed_partial)

block_types = [
    Block(name, globals()[f'build_{name}'], allowed_neighbors=allowed_neighbors[name])
    for name in path_blocks
]

# --- Run WFC and build scene ---
wfc = WaveFunctionCollapse3D(6, 2, 3, block_types)  # 6x3x6 grid for demo
wfc.collapse()  # Seed for reproducibility
wfc.build_scene(scene)
scene.finish()
