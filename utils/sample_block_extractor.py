import random
import numpy as np
from typing import Tuple, List, Dict
from collections import defaultdict
from wfc import Block

class SampleBlockExtractor:
    def __init__(self, sample_scene: np.ndarray, block_shape: Tuple[int, int, int], similarity_threshold: float = 0.95, neighbor_distance: int = 0, material_compatibility_map: Dict[frozenset, float] = None, allow_repeated_blocks: bool = False):
        """
        sample_scene: (A, B, C, 4) numpy array
        block_shape: (Nx, Ny, Nz)
        similarity_threshold: threshold for average cosine similarity to consider two block faces connectable
        neighbor_distance: distance to consider for neighbor connections
        material_compatibility_map: dict mapping frozenset({mat1, mat2}) -> float in [0,1] for material compatibility
        allow_repeated_blocks: if True, blocks are not required to be unique (can repeat)
        """
        self.sample_scene = sample_scene
        self.block_shape = block_shape
        self.similarity_threshold = similarity_threshold
        self.neighbor_distance = neighbor_distance
        self.allow_repeated_blocks = allow_repeated_blocks
        self.blocks = []  # List of blocks (as np.ndarray)
        self.block_indices = {}  # Map from block hash to index in self.blocks
        self.allowed_neighbors = defaultdict(lambda: defaultdict(set))  # block_idx -> direction -> set(block_idx)
        self.block_origins = []  # Store the origin of each block in the sample
        if material_compatibility_map is None:
            # By default, air (0) matches with light (2) and block (1) with 1.0
            self.material_compatibility_map = {
                frozenset([0, 0]): 1.0,
                frozenset([0, 1]): 1.0,
                frozenset([0, 2]): 1.0,
                frozenset([1, 1]): 1.0,
                frozenset([2, 2]): 1.0,
                frozenset([1, 2]): 0.0,
            }
        else:
            # Convert all keys to frozenset for safety
            self.material_compatibility_map = {frozenset(k): v for k, v in material_compatibility_map.items()}
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
                    if self.allow_repeated_blocks:
                        # Always add the block, even if identical
                        idx = len(self.blocks)
                        self.blocks.append(block)
                        self.block_origins.append((i, j, k))
                        block_map[(i, j, k)] = idx
                    else:
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

    def _get_block_faces_for_connection(self, block1: np.ndarray, block2: np.ndarray, direction: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to get the opposing faces of two blocks based on direction."""
        if direction == (1, 0, 0): # block1 is to the left of block2
            face1 = block1[-1, :, :, :] # Right face of block1
            face2 = block2[0, :, :, :]  # Left face of block2
        elif direction == (-1, 0, 0): # block1 is to the right of block2
            face1 = block1[0, :, :, :]  # Left face of block1
            face2 = block2[-1, :, :, :] # Right face of block2
        elif direction == (0, 1, 0): # block1 is below block2
            face1 = block1[:, -1, :, :] # Top face of block1
            face2 = block2[:, 0, :, :]  # Bottom face of block2
        elif direction == (0, -1, 0): # block1 is above block2
            face1 = block1[:, 0, :, :]  # Bottom face of block1
            face2 = block2[:, -1, :, :] # Top face of block2
        elif direction == (0, 0, 1): # block1 is behind block2
            face1 = block1[:, :, -1, :] # Front face of block1
            face2 = block2[:, :, 0, :]  # Back face of block2
        elif direction == (0, 0, -1): # block1 is in front of block2
            face1 = block1[:, :, 0, :]  # Back face of block1
            face2 = block2[:, :, -1, :] # Front face of block2
        else:
            raise ValueError(f"Invalid direction for face extraction: {direction}")
        return face1, face2

    def _get_block_region_for_connection(self, block: np.ndarray, direction: Tuple[int, int, int], n: int) -> np.ndarray:
        """Helper to get a region of thickness n from a block, corresponding to an incoming connection from 'direction'."""
        if direction == (1, 0, 0): # Connection comes from left (-x), so take left region of block
            region = block[:n, :, :, :]
        elif direction == (-1, 0, 0): # Connection comes from right (+x), so take right region of block
            region = block[-n:, :, :, :]
        elif direction == (0, 1, 0): # Connection comes from bottom (-y), so take bottom region of block
            region = block[:, :n, :, :]
        elif direction == (0, -1, 0): # Connection comes from top (+y), so take top region of block
            region = block[:, -n:, :, :]
        elif direction == (0, 0, 1): # Connection comes from back (-z), so take back region of block
            region = block[:, :, :n, :]
        elif direction == (0, 0, -1): # Connection comes from front (+z), so take front region of block
            region = block[:, :, -n:, :]
        else:
            raise ValueError(f"Invalid direction for region extraction from block: {direction}")
        return region

    def _compare_faces_or_regions(self, arr1: np.ndarray, arr2: np.ndarray) -> bool:
        """
        Compare two faces or regions for connectability based on material and color similarity.
        Returns True if connectable, False otherwise.
        Uses material_compatibility_map for material matching.
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
        if mat1.size == 0 or mat2.size == 0: # It's occurs on the edges of the sample
            return False
        similarities = []
        for idx, (m1, m2) in enumerate(zip(mat1, mat2)):
            key = frozenset([int(m1), int(m2)])
            compat = self.material_compatibility_map.get(key, 0.0)
            norm1 = np.linalg.norm(color1[idx])
            norm2 = np.linalg.norm(color2[idx])
            if norm1 == 0 or norm2 == 0:
                cos_sim = 1.0 # Treat zero vectors as identical
            else:
                cos_sim = np.dot(color1[idx]/norm1, color2[idx]/norm2)
            combined_sim = compat * cos_sim
            similarities.append(combined_sim)
        avg_similarity = np.mean(similarities)
        return avg_similarity >= self.similarity_threshold

    def _blocks_can_connect(self, idx1: int, idx2: int, direction: Tuple[int, int, int]) -> bool:
        block1 = self.blocks[idx1]
        block2 = self.blocks[idx2]

        arr1_to_compare = None
        arr2_to_compare = None
        attempt_region_comparison = False

        if self.neighbor_distance > 0:
            n = self.neighbor_distance
            block1_origin = self.block_origins[idx1]
            
            # _get_sample_neighbor_region is expected to return an empty np.ndarray 
            # if the requested region is out of bounds of the sample_scene.
            # It raises ValueError for fundamentally invalid direction tuples, 
            # but directions from _extract_blocks_and_connections are assumed valid.
            region1_from_sample = self._get_sample_neighbor_region(block1_origin, direction, n)

            if region1_from_sample.size > 0:
                # Successfully obtained a non-empty region from the sample scene
                arr1_to_compare = region1_from_sample
                arr2_to_compare = self._get_block_region_for_connection(block2, direction, n)
                attempt_region_comparison = True
            # If region1_from_sample.size == 0 (edge case), we fall through to face comparison.

        if not attempt_region_comparison:
            # This block is executed if:
            # 1. self.neighbor_distance == 0 (initial condition)
            # 2. self.neighbor_distance > 0 BUT region1_from_sample was empty (edge of scene)
            arr1_to_compare, arr2_to_compare = self._get_block_faces_for_connection(block1, block2, direction)

        return self._compare_faces_or_regions(arr1_to_compare, arr2_to_compare)

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







