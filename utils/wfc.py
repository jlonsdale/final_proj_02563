import taichi as ti
import numpy as np
from taichi.math import vec3, ivec3

@ti.kernel
def build_kernel(scene: ti.template(), x: int, y: int, z: int, data: ti.types.ndarray()):
    sx, sy, sz, _ = data.shape
    for i, j, k in ti.ndrange(sx, sy, sz):
        r = data[i, j, k, 0]
        g = data[i, j, k, 1]
        b = data[i, j, k, 2]
        mat = int(data[i, j, k, 3])
        if mat != 0:
            scene.set_voxel(ivec3(x + i, y + j, z + k), mat, vec3(r, g, b))
            
# --- Block class ---
class Block:
    def __init__(self, name, data, allowed_neighbors=None, metadata=None, weight=1.0):
        self.name = name
        # data: numpy array of shape (3,3,3,4) with (r,g,b,mat)
        self.data = data.astype(np.float32)
        self.allowed_neighbors = allowed_neighbors or {}
        self.metadata = metadata or {}
        self.weight = weight

    def build(self, scene, pos):
        build_kernel(scene, pos[0], pos[1], pos[2], self.data)

# --- WFC 3D class ---
class WaveFunctionCollapse3D:
    def __init__(self, width, height, depth, block_types, seed=None, enforce_ground_constraint=False):
        self.width = width
        self.height = height
        self.depth = depth
        self.block_types = block_types  # List of Block objects
        self.block_types_by_name = {b.name: b for b in block_types}
        self.grid = np.full((width, height, depth), None)  # Collapsed block at each cell
        self.enforce_ground_constraint = enforce_ground_constraint
        # Store sets of block names instead of Block objects
        self.possible_blocks = []
        for x in range(width):
            col = []
            for y in range(height):
                row = []
                for z in range(depth):
                    if self.enforce_ground_constraint and y == 0:
                        # Only allow blocks with can_be_ground=True at ground level
                        allowed = set(
                            b.name for b in block_types if b.metadata.get('can_be_ground', False)
                        )
                    else:
                        allowed = set(self.block_types_by_name.keys())
                    row.append(allowed)
                col.append(row)
            self.possible_blocks.append(col)
        # Determine block shape from the first block
        if len(block_types) > 0:
            self.block_shape = block_types[0].data.shape[:3]
        else:
            self.block_shape = (1, 1, 1)
        self.rng = np.random.default_rng(seed)

    def collapse(self):
        # Collapse cells in top-to-bottom, left-to-right, front-to-back order
        for y in range(self.height):
            for x in range(self.width):
                for z in range(self.depth):
                    if self.grid[x, y, z] is None:
                        options = self.possible_blocks[x][y][z]
                        if not options:
                            continue  # No options left, skip (or could raise error)
                        option_names = sorted(list(options))
                        option_blocks = [self.block_types_by_name[name] for name in option_names]
                        weights = [b.weight for b in option_blocks]
                        weights = np.array(weights, dtype=np.float32)
                        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
                        chosen_name = self.rng.choice(option_names, p=weights)
                        chosen = self.block_types_by_name[chosen_name]
                        self.grid[x, y, z] = chosen
                        self.possible_blocks[x][y][z] = {chosen_name}
                        self.propagate(x, y, z)

    def propagate(self, x, y, z):
        # Stack-based propagation: propagate constraints until no more changes
        stack = [(x, y, z)]
        directions = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        while stack:
            cx, cy, cz = stack.pop()
            for dx, dy, dz in directions:
                nx, ny, nz = cx+dx, cy+dy, cz+dz
                if 0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth and self.grid[nx, ny, nz] is None:
                    if len(self.possible_blocks[cx][cy][cz]) == 0:
                        # No options left, this should not 
                        continue
                        raise ValueError("No options left for cell ({}, {}, {})".format(cx, cy, cz))
                    allowed_names = set()
                    for block_name in self.possible_blocks[cx][cy][cz]:
                        allowed_names |= set(self.block_types_by_name[block_name].allowed_neighbors.get((dx, dy, dz), []))
                    before = self.possible_blocks[nx][ny][nz]
                    self.possible_blocks[nx][ny][nz] = self.possible_blocks[nx][ny][nz].intersection(allowed_names)
                    after = self.possible_blocks[nx][ny][nz]
                    if len(after) < len(before) and len(after) != 0:
                        stack.append((nx, ny, nz))

    def build_scene(self, scene):
        # print the grind
        print("Grid:")
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    print(f"({x}, {y}, {z}): {self.grid[x, y, z].name if self.grid[x, y, z] else 'None'}")
        bx, by, bz = self.block_shape
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    block = self.grid[x, y, z]
                    if block:
                        block.build(scene, (bx*x, by*y, bz*z))

    def save_scene_as_ndarray(self, filename):
        """
        Save the current WFC grid as a numpy ndarray of voxel values.
        The output shape is (width, height, depth, bx, by, bz, 4), where (bx, by, bz) is the block shape.
        """
        bx, by, bz = self.block_shape
        arr = np.zeros((self.width, self.height, self.depth, bx, by, bz, 4), dtype=np.float32)
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    block = self.grid[x, y, z]
                    if block is not None:
                        arr[x, y, z] = block.data
        np.save(filename, arr)

    @staticmethod
    def load_scene_from_ndarray(filename):
        """
        Load a WFC scene ndarray from file. Returns the ndarray.
        """
        return np.load(filename)

