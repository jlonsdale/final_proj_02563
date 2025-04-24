import taichi as ti
import numpy as np
from taichi.math import vec3, ivec3

# --- Block class ---
class Block:
    def __init__(self, name, recipe_func, allowed_neighbors=None):
        self.name = name
        self.recipe_func = recipe_func  # Function to build this block in the scene
        # allowed_neighbors: dict of (dx, dy, dz) -> list of allowed block names
        self.allowed_neighbors = allowed_neighbors or {}

    def build(self, scene, pos):
        self.recipe_func(scene, pos[0], pos[1], pos[2])

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
        # # print the grind
        # print("Grid:")
        # for x in range(self.width):
        #     for y in range(self.height):
        #         for z in range(self.depth):
        #             print(f"({x}, {y}, {z}): {self.grid[x, y, z].name if self.grid[x, y, z] else 'None'}")
        
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    block = self.grid[x, y, z]
                    if block:
                        block.build(scene, (x, y, z))

# --- Example usage ---
# from scene import Scene
# scene = Scene(...)
# block_types = [Block('grass', vec3(0.03, 0.45, 0.03), grass_block_recipe), ...]
# wfc = WaveFunctionCollapse3D(width, height, depth, block_types)
# wfc.collapse()
# wfc.build_scene(scene)

# Extend Block and WaveFunctionCollapse3D for more features as needed.