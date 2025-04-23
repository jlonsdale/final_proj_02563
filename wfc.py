import taichi as ti
import numpy as np
from taichi.math import vec3, ivec3

# --- Block class ---
class Block:
    def __init__(self, name, recipe_func, allowed_neighbors=None):
        self.name = name
        self.recipe_func = recipe_func  # Function to build this block in the scene
        self.allowed_neighbors = allowed_neighbors or {}

    def build(self, scene, pos):
        # Call the kernel function directly with scene, x, y, z, color
        self.recipe_func(scene, pos[0], pos[1], pos[2])

# --- WFC 3D class (MVP: 2D layer) ---
class WaveFunctionCollapse3D:
    def __init__(self, width, height, block_types):
        self.width = width
        self.height = height
        self.block_types = block_types  # List of Block objects
        self.grid = np.full((width, height), None)  # Collapsed block at each cell
        self.possible_blocks = [[set(block_types) for _ in range(height)] for _ in range(width)]

    def collapse(self):
        # MVP: Randomly pick a cell with lowest entropy and collapse
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] is None:
                    options = self.possible_blocks[x][y]
                    if options:
                        chosen = np.random.choice(list(options))
                        self.grid[x, y] = chosen
                        self.propagate(x, y, chosen)
        # (This is a naive MVP, not a full WFC implementation)

    def propagate(self, x, y, chosen_block):
        # MVP: Remove incompatible blocks from neighbors
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[nx, ny] is None:
                allowed = chosen_block.allowed_neighbors.get((dx, dy), None)
                if allowed is not None:
                    self.possible_blocks[nx][ny] = self.possible_blocks[nx][ny].intersection(allowed)

    def build_scene(self, scene):
        for x in range(self.width):
            for y in range(self.height):
                block = self.grid[x, y]
                if block:
                    block.build(scene, (x, 0, y))



# --- Example usage ---
# from scene import Scene
# scene = Scene(...)
# block_types = [Block('grass', vec3(0.03, 0.45, 0.03), grass_block_recipe), ...]
# wfc = WaveFunctionCollapse3D(width, height, block_types)
# wfc.collapse()
# wfc.build_scene(scene)

# Extend Block and WaveFunctionCollapse3D for more features as needed.