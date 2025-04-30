import random
import numpy as np
from typing import Tuple, List, Dict
from collections import defaultdict
from wfc import Block

class SampleBlockExtractor:
    def __init__(self, sample_scene: np.ndarray, block_shape: Tuple[int, int, int], similarity_threshold: float = 0.95):
        """
        sample_scene: (A, B, C, 4) numpy array
        block_shape: (Nx, Ny, Nz)
        similarity_threshold: threshold for average cosine similarity to consider two block faces connectable
        """
        self.sample_scene = sample_scene
        self.block_shape = block_shape
        self.similarity_threshold = similarity_threshold
        self.blocks = []  # List of unique blocks (as np.ndarray)
        self.block_indices = {}  # Map from block hash to index in self.blocks
        self.allowed_neighbors = defaultdict(lambda: defaultdict(set))  # block_idx -> direction -> set(block_idx)
        self._extract_blocks_and_connections()

    def _extract_blocks_and_connections(self):
        A, B, C, _ = self.sample_scene.shape
        Nx, Ny, Nz = self.block_shape
        block_map = {}
        for i in range(A - Nx + 1):
            for j in range(B - Ny + 1):
                for k in range(C - Nz + 1):
                    block = self.sample_scene[i:i+Nx, j:j+Ny, k:k+Nz, :].copy()
                    block_hash = self._hash_block(block)
                    if block_hash not in self.block_indices:
                        self.block_indices[block_hash] = len(self.blocks)
                        self.blocks.append(block)
                    block_map[(i, j, k)] = self.block_indices[block_hash]
        # Now, infer connections
        directions = [
            (1, 0, 0),  # +x (east)
            (-1, 0, 0), # -x (west)
            (0, 1, 0),  # +y (up)
            (0, -1, 0), # -y (down)
            (0, 0, 1),  # +z (south)
            (0, 0, -1), # -z (north)
        ]
        for (i, j, k), idx in block_map.items():
            for d, (dx, dy, dz) in enumerate(directions):
                ni, nj, nk = i + dx, j + dy, k + dz
                if (ni, nj, nk) in block_map:
                    idx2 = block_map[(ni, nj, nk)]
                    if self._blocks_can_connect(self.blocks[idx], self.blocks[idx2], (dx, dy, dz)):
                        self.allowed_neighbors[idx][(dx, dy, dz)].add(idx2)

    def _hash_block(self, block: np.ndarray) -> int:
        # Use a hash of the bytes for uniqueness
        return hash(block.tobytes())  # Uncomment for a real hash

    def _blocks_can_connect(self, block1: np.ndarray, block2: np.ndarray, direction: Tuple[int, int, int]) -> bool:
        # Compare the face of block1 and the opposite face of block2 along the direction
        # Faces are (Nx, Ny, Nz, 4)
        if direction == (1, 0, 0):  # block1 +x face, block2 -x face
            face1 = block1[-1, :, :, :]
            face2 = block2[0, :, :, :]
        elif direction == (-1, 0, 0):
            face1 = block1[0, :, :, :]
            face2 = block2[-1, :, :, :]
        elif direction == (0, 1, 0):
            face1 = block1[:, -1, :, :]
            face2 = block2[:, 0, :, :]
        elif direction == (0, -1, 0):
            face1 = block1[:, 0, :, :]
            face2 = block2[:, -1, :, :]
        elif direction == (0, 0, 1):
            face1 = block1[:, :, -1, :]
            face2 = block2[:, :, 0, :]
        elif direction == (0, 0, -1):
            face1 = block1[:, :, 0, :]
            face2 = block2[:, :, -1, :]
        else:
            raise ValueError("Invalid direction")
        # Only compare where material matches
        mat1 = face1[..., 3].reshape(-1)
        mat2 = face2[..., 3].reshape(-1)
        color1 = face1[..., :3].reshape(-1, 3)
        color2 = face2[..., :3].reshape(-1, 3)
        # Find indices where material matches
        mat_mask = (mat1 == mat2)
        if np.all((mat1 == 0) & (mat2 == 0)):
            # Both faces are air, treat as perfect match
            return True
        if not np.any(mat_mask):
            # No matching material, cannot connect
            return False
        # Only compare color where material matches
        f1 = color1[mat_mask]
        f2 = color2[mat_mask]
        # Avoid division by zero
        norm1 = np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8
        norm2 = np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8
        f1n = f1 / norm1
        f2n = f2 / norm2
        cos_sim = np.sum(f1n * f2n, axis=1)
        avg_sim = np.mean(cos_sim)
        return avg_sim >= self.similarity_threshold

    def get_blocks(self) -> List[np.ndarray]:
        return self.blocks

    def get_allowed_neighbors(self) -> Dict[int, Dict[Tuple[int, int, int], List[int]]]:
        # Convert sets to lists for easier use
        return {idx: {d: list(neighs) for d, neighs in dirs.items()} for idx, dirs in self.allowed_neighbors.items()}

    def get_block_objects(self):
        """
        Returns a list of Block objects (from wfc.py) with unique names and proper allowed_neighbors (by index).
        """
        block_objects = []
        allowed_neighbors = self.get_allowed_neighbors()
        for i, block_data in enumerate(self.blocks):
            name = f"block_{i}"
            neighbors = allowed_neighbors.get(i, {})
            block_objects.append(Block(name, block_data, allowed_neighbors=neighbors))
        return block_objects


# --- DEMO: Extract blocks from a sample scene ---
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

def make_sample_scene_with_blocks():
        # Create a 4x4x4 scene with 4 channels (r,g,b,mat)
    arr = np.zeros((4, 4, 4, 4), dtype=np.float32)
    # Fill with colored cubes to cover the area
    arr[0:2, 0:2, 0:2, 0:3] = (1.0, 0.0, 0.0)  # Red cube
    arr[0:2, 0:2, 2:4, 0:3] = (0.0, 1.0, 0.0)  # Green cube
    arr[2:4, 0:2, 0:2, 0:3] = (0.0, 0.0, 1.0)  # Blue cube
    arr[2:4, 2:4, 2:4, 0:3] = (1.0, 1.0, 0.0)  # Yellow cube
    arr[0:2, 2:4, 0:2, 0:3] = (1.0, 0.0, 1.0)  # Magenta cube
    arr[2:4, 0:2, 2:4, 0:3] = (0.0, 1.0, 1.0)  # Cyan cube
    arr[0:2, 2:4, 2:4, 0:3] = (0.5, 0.5, 0.5)  # Gray cube
    arr[2:4, 2:4, 0:2, 0:3] = (1.0, 0.5, 0.0)  # Orange cube
    arr[..., 3] = 1  # Set material to 1 everywhere
    return arr

if __name__ == "__main__":
    from scene import Scene
    sample_scene = make_sample_scene_with_blocks()
    block_shape = (2, 2, 2)
    extractor = SampleBlockExtractor(sample_scene, block_shape, similarity_threshold=0.99)
    block_objects = extractor.get_block_objects()
    print(f"Extracted {len(block_objects)} unique blocks.")
    scene = Scene(voxel_edges=0, exposure=1)
    scene.set_floor(0, (1.0, 1.0, 1.0))
    scene.set_background_color((0.5, 0.5, 0.4))
    scene.set_directional_light((1, 1, 1), 0.1, (1, 0.8, 0.6))
    from wfc import build_kernel
    # Visualize the original sample scene at the origin
    build_kernel(scene, 0, 0, 0, sample_scene)
    # Visualize all blocks in the scene, spaced apart
    for i, block in enumerate(block_objects):
        base_pos = ((i - 9) * (block_shape[0] + 1), 0, -5)
        block.build(scene, base_pos)
        # Build all possible neighbors in all directions, stacking them vertically
        stack_y = 1
        for direction, neighbor_indices in block.allowed_neighbors.items():
            print(f"Building neighbors for block {i} in direction {direction}: {neighbor_indices}")
            for neighbor_idx in neighbor_indices:
                neighbor_block = block_objects[neighbor_idx]
                neighbor_pos = (base_pos[0], base_pos[1] + stack_y * (block_shape[1] + 1), base_pos[2])
                neighbor_block.build(scene, neighbor_pos)
                stack_y += 1
    scene.finish()

