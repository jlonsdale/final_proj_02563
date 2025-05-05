import random
import numpy as np
from typing import Tuple, List, Dict
from collections import defaultdict
from wfc import Block

class SampleBlockExtractor:
    def __init__(self, sample_scene: np.ndarray, block_shape: Tuple[int, int, int], similarity_threshold: float = 0.95, neighbor_distance: int = 0):
        """
        sample_scene: (A, B, C, 4) numpy array
        block_shape: (Nx, Ny, Nz)
        similarity_threshold: threshold for average cosine similarity to consider two block faces connectable
        neighbor_distance: distance to consider for neighbor connections
        """
        self.sample_scene = sample_scene
        self.block_shape = block_shape
        self.similarity_threshold = similarity_threshold
        self.neighbor_distance = neighbor_distance
        self.blocks = []  # List of unique blocks (as np.ndarray)
        self.block_indices = {}  # Map from block hash to index in self.blocks
        self.allowed_neighbors = defaultdict(lambda: defaultdict(set))  # block_idx -> direction -> set(block_idx)
        self.block_origins = []  # Store the origin of each block in the sample
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
                        self.block_origins.append((i, j, k))
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
        # 1. Add neighbors from the sample
        for (i, j, k), idx in block_map.items():
            for d, (dx, dy, dz) in enumerate(directions):
                ni, nj, nk = i + dx, j + dy, k + dz
                if (ni, nj, nk) in block_map:
                    idx2 = block_map[(ni, nj, nk)]
                    if self._blocks_can_connect(idx, idx2, (dx, dy, dz)):
                        self.allowed_neighbors[idx][(dx, dy, dz)].add(idx2)
        # 2. For each block, check all possible connections in all directions
        num_blocks = len(self.blocks)
        for idx1 in range(num_blocks):
            for d, direction in enumerate(directions):
                for idx2 in range(num_blocks):
                    if self._blocks_can_connect(idx1, idx2, direction):
                        self.allowed_neighbors[idx1][direction].add(idx2)

    def _get_sample_neighbor_region(self, block_origin, direction, n):
        # Returns the region of the sample_scene that was adjacent to the block at block_origin in direction, thickness n
        i, j, k = block_origin
        Nx, Ny, Nz = self.block_shape
        if direction == (1, 0, 0):
            region = self.sample_scene[i+Nx:i+Nx+n, j:j+Ny, k:k+Nz, :]
        elif direction == (-1, 0, 0):
            region = self.sample_scene[i-n:i, j:j+Ny, k:k+Nz, :]
        elif direction == (0, 1, 0):
            region = self.sample_scene[i:i+Nx, j+Ny:j+Ny+n, k:k+Nz, :]
        elif direction == (0, -1, 0):
            region = self.sample_scene[i:i+Nx, j-n:j, k:k+Nz, :]
        elif direction == (0, 0, 1):
            region = self.sample_scene[i:i+Nx, j:j+Ny, k+Nz:k+Nz+n, :]
        elif direction == (0, 0, -1):
            region = self.sample_scene[i:i+Nx, j:j+Ny, k-n:k, :]
        else:
            raise ValueError("Invalid direction")
        return region

    def _compare_faces_or_regions(self, arr1: np.ndarray, arr2: np.ndarray) -> bool:
        """
        Compare two faces or regions for connectability based on material and color similarity.
        Returns True if connectable, False otherwise.
        """
        if arr1.shape != arr2.shape:
            # resize the the biggest one to the smallest one
            min_shape = np.minimum(arr1.shape, arr2.shape)
            arr1 = arr1[tuple(slice(0, s) for s in min_shape)]
            arr2 = arr2[tuple(slice(0, s) for s in min_shape)]
        mat1 = arr1[..., 3].reshape(-1)
        mat2 = arr2[..., 3].reshape(-1)
        color1 = arr1[..., :3].reshape(-1, 3)
        color2 = arr2[..., :3].reshape(-1, 3)
        # here is one of the matrices is empty we will have a match. The array can be empty if block is on the edge of the scene
        if np.all(mat1 == 0) or np.all(mat2 == 0):
            return True
        mat_mask = (mat1 == mat2)
        mat_mask = np.logical_xor(mat_mask, (mat1 == 0) & (mat2 == 0))
        if not np.any(mat_mask):
            return False
        f1 = color1[mat_mask]
        f2 = color2[mat_mask]
        norm1 = np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8
        norm2 = np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8
        f1n = f1 / norm1
        f2n = f2 / norm2
        cos_sim = np.sum(f1n * f2n, axis=1)
        avg_sim = np.mean(cos_sim)
        return avg_sim >= self.similarity_threshold

    def _blocks_can_connect(self, idx1: int, idx2: int, direction: Tuple[int, int, int]) -> bool:
        block1 = self.blocks[idx1]
        block2 = self.blocks[idx2]
        if self.neighbor_distance == 0:
            if direction == (1, 0, 0):
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
            return self._compare_faces_or_regions(face1, face2)
        else:
            n = self.neighbor_distance
            block1_origin = self.block_origins[idx1]
            region1 = self._get_sample_neighbor_region(block1_origin, direction, n)
            if direction == (1, 0, 0):
                region2 = block2[:n, :, :, :]
            elif direction == (-1, 0, 0):
                region2 = block2[-n:, :, :, :]
            elif direction == (0, 1, 0):
                region2 = block2[:, :n, :, :]
            elif direction == (0, -1, 0):
                region2 = block2[:, -n:, :, :]
            elif direction == (0, 0, 1):
                region2 = block2[:, :, :n, :]
            elif direction == (0, 0, -1):
                region2 = block2[:, :, -n:, :]
            else:
                raise ValueError("Invalid direction")
            return self._compare_faces_or_regions(region1, region2)

    def _hash_block(self, block: np.ndarray) -> int:
        # Use a hash of the bytes for uniqueness
        return hash(block.tobytes())

    def get_blocks(self) -> List[np.ndarray]:
        return self.blocks

    def get_allowed_neighbors(self) -> Dict[int, Dict[Tuple[int, int, int], List[int]]]:
        # Convert sets to lists for easier use
        return {idx: {d: list(neighs) for d, neighs in dirs.items()} for idx, dirs in self.allowed_neighbors.items()}

    def get_block_objects(self):
        """
        Returns a list of Block objects (from wfc.py) with unique names and allowed_neighbors using block names.
        """
        idx_to_name = {i: f"block_{i}" for i in range(len(self.blocks))}
        allowed_neighbors = self.get_allowed_neighbors()
        block_objects = []
        for idx, block_data in enumerate(self.blocks):
            name = idx_to_name[idx]
            neighbors = allowed_neighbors.get(idx, {})
            # Map neighbor indices to names for each direction
            neighbors_named = {
                direction: [idx_to_name[n_idx] for n_idx in neighbor_indices]
                for direction, neighbor_indices in neighbors.items()
            }
            block_objects.append(Block(name, block_data, allowed_neighbors=neighbors_named))
        return block_objects







