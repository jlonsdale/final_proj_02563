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
    def __init__(self, name, data, allowed_neighbors=None):
        self.name = name
        # data: numpy array of shape (3,3,3,4) with (r,g,b,mat)
        self.data = data.astype(np.float32)
        self.allowed_neighbors = allowed_neighbors or {}

    def build(self, scene, pos):
        build_kernel(scene, pos[0], pos[1], pos[2], self.data)

# --- WFC 3D class ---
class WaveFunctionCollapse3D:
    def __init__(self, width, height, depth, block_types):
        self.width = width
        self.height = height
        self.depth = depth
        self.block_types = block_types  # List of Block objects
        self.block_types_by_name = {b.name: b for b in block_types}
        self.grid = np.full((width, height, depth), None)  # Collapsed block at each cell
        # Store sets of block names instead of Block objects
        self.possible_blocks = [[[set(self.block_types_by_name.keys()) for _ in range(depth)] for _ in range(height)] for _ in range(width)]
        # Determine block shape from the first block
        if len(block_types) > 0:
            self.block_shape = block_types[0].data.shape[:3]
        else:
            self.block_shape = (1, 1, 1)

    def collapse(self):
        # MVP: Randomly pick a cell with lowest entropy and collapse
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    if self.grid[x, y, z] is None:
                        options = self.possible_blocks[x][y][z]
                        if options:
                            chosen_name = np.random.choice(list(options))
                            chosen = self.block_types_by_name[chosen_name]
                            self.grid[x, y, z] = chosen
                            self.possible_blocks[x][y][z] = {chosen_name}
                            self.propagate(x, y, z)
        # (This is a naive MVP, not a full WFC implementation)

    def propagate(self, x, y, z):
        # Stack-based propagation: propagate constraints until no more changes
        stack = [(x, y, z)]
        directions = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        while stack:
            cx, cy, cz = stack.pop()
            for dx, dy, dz in directions:
                nx, ny, nz = cx+dx, cy+dy, cz+dz
                if 0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth and self.grid[nx, ny, nz] is None:
                    allowed_names = set()
                    for block_name in self.possible_blocks[cx][cy][cz]:
                        allowed_names |= set(self.block_types_by_name[block_name].allowed_neighbors.get((dx, dy, dz), []))
                    if allowed_names:
                        before = self.possible_blocks[nx][ny][nz]
                        self.possible_blocks[nx][ny][nz] = self.possible_blocks[nx][ny][nz].intersection(allowed_names)
                        after = self.possible_blocks[nx][ny][nz]
                        if len(after) < len(before):
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

if __name__ == "__main__":
    import sample_block_extractor
    from scene import Scene
    
    # Create a sample scene
    sample_scene = sample_block_extractor.make_sample_scene()
    block_shape = (2, 2, 2)
    extractor = sample_block_extractor.SampleBlockExtractor(sample_scene, block_shape, similarity_threshold=0.99)
    block_objects = extractor.get_block_objects()
    print(f"Extracted {len(block_objects)} unique blocks.")
    scene = Scene(voxel_edges=0, exposure=1)
    scene.set_floor(0, (1.0, 1.0, 1.0))
    scene.set_background_color((0.5, 0.5, 0.4))
    scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6))
    
    wfc = WaveFunctionCollapse3D(10, 10, 10, block_objects)
    wfc.collapse()
    wfc.build_scene(scene)
    scene.finish()