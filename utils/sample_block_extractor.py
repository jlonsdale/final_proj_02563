import random
import numpy as np
from typing import Tuple, List, Dict
from collections import defaultdict
from wfc import Block
import sys
import json
import os

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

def check_connections_for_block(args):
    idx1, block_faces, face_hash_map, directions, blocks, block_origins, neighbor_distance, similarity_threshold, material_compatibility_map = args
    from collections import defaultdict

    local_allowed = defaultdict(set)
    for d in directions:
        face1 = block_faces[d][idx1]
        face_hash = hash(face1.tobytes())
        candidate_indices = face_hash_map[d][face_hash]
        for idx2 in candidate_indices:
            block1 = blocks[idx1]
            block2 = blocks[idx2]
            arr1_to_compare = None
            arr2_to_compare = None
            attempt_region_comparison = False
            if neighbor_distance > 0:
                n = neighbor_distance
                block1_origin = block_origins[idx1]
            if not attempt_region_comparison:
                if d == (1, 0, 0):
                    arr1_to_compare = block1[-1, :, :, :]
                    arr2_to_compare = block2[0, :, :, :]
                elif d == (-1, 0, 0):
                    arr1_to_compare = block1[0, :, :, :]
                    arr2_to_compare = block2[-1, :, :, :]
                elif d == (0, 1, 0):
                    arr1_to_compare = block1[:, -1, :, :]
                    arr2_to_compare = block2[:, 0, :, :]
                elif d == (0, -1, 0):
                    arr1_to_compare = block1[:, 0, :, :]
                    arr2_to_compare = block2[:, -1, :, :]
                elif d == (0, 0, 1):
                    arr1_to_compare = block1[:, :, -1, :]
                    arr2_to_compare = block2[:, :, 0, :]
                elif d == (0, 0, -1):
                    arr1_to_compare = block1[:, :, 0, :]
                    arr2_to_compare = block2[:, :, -1, :]
            arr1 = arr1_to_compare
            arr2 = arr2_to_compare
            if arr1.shape != arr2.shape:
                min_shape = np.minimum(arr1.shape, arr2.shape)
                arr1 = arr1[tuple(slice(0, s) for s in min_shape)]
                arr2 = arr2[tuple(slice(0, s) for s in min_shape)]
            mat1 = arr1[..., 3].reshape(-1)
            mat2 = arr2[..., 3].reshape(-1)
            color1 = arr1[..., :3].reshape(-1, 3)
            color2 = arr2[..., :3].reshape(-1, 3)
            if mat1.size == 0 or mat2.size == 0:
                continue
            similarities = []
            for idx, (m1, m2) in enumerate(zip(mat1, mat2)):
                key = frozenset([int(m1), int(m2)])
                compat = material_compatibility_map.get(key, 0.0)
                norm1 = np.linalg.norm(color1[idx])
                norm2 = np.linalg.norm(color2[idx])
                if norm1 == 0 or norm2 == 0:
                    cos_sim = 1.0
                else:
                    cos_sim = np.dot(color1[idx]/norm1, color2[idx]/norm2)
                combined_sim = compat * cos_sim
                similarities.append(combined_sim)
            avg_similarity = np.mean(similarities)
            if avg_similarity >= similarity_threshold:
                local_allowed[d].add(idx2)
    return idx1, local_allowed

class SampleBlockExtractor:
    def __init__(self, sample_scene: np.ndarray, block_shape: Tuple[int, int, int], similarity_threshold: float = 0.95, neighbor_distance: int = 0, material_compatibility_map: Dict[frozenset, float] = None, allow_repeated_blocks: bool = False):
        self.sample_scene = sample_scene
        self.block_shape = block_shape
        self.similarity_threshold = similarity_threshold
        self.neighbor_distance = neighbor_distance
        self.allow_repeated_blocks = allow_repeated_blocks
        self.blocks = []  # List of blocks (as np.ndarray)
        self.block_indices = {}  # Map from block hash to index in self.blocks
        self.allowed_neighbors = defaultdict(lambda: defaultdict(set))  # block_idx -> direction -> set(block_idx)
        self.block_origins = []  # Store the origin of each block in the sample
        self.block_count_by_index = defaultdict(int)  # Count of blocks by index
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
            self.material_compatibility_map = {frozenset(k): v for k, v in material_compatibility_map.items()}
        self._extract_blocks_and_connections()

    def _extract_blocks_and_connections(self):
        import concurrent.futures
        A, B, C, _ = self.sample_scene.shape
        Nx, Ny, Nz = self.block_shape
        block_map = {}
        total_blocks = (A - Nx + 1) * (B - Ny + 1) * (C - Nz + 1)
        block_iter = (
            tqdm(range(A - Nx + 1), desc='Extracting blocks', file=sys.stdout)
            if _TQDM_AVAILABLE else range(A - Nx + 1)
        )
        block_counter = 0
        for i in block_iter:
            for j in range(B - Ny + 1):
                for k in range(C - Nz + 1):
                    block = self.sample_scene[i:i+Nx, j:j+Ny, k:k+Nz, :].copy()
                    block_hash = self._hash_block(block)
                    if self.allow_repeated_blocks:
                        idx = len(self.blocks)
                        self.blocks.append(block)
                        self.block_origins.append((i, j, k))
                        block_map[(i, j, k)] = idx
                        self.block_count_by_index[idx] += 1
                    else:
                        if block_hash not in self.block_indices:
                            self.block_indices[block_hash] = len(self.blocks)
                            self.blocks.append(block)
                            self.block_origins.append((i, j, k))
                        block_map[(i, j, k)] = self.block_indices[block_hash]
                        self.block_count_by_index[self.block_indices[block_hash]] += 1
                    block_counter += 1
                    if not _TQDM_AVAILABLE and block_counter % 1000 == 0:
                        print(f"Extracted {block_counter}/{total_blocks} blocks...")
        directions = [
            (1, 0, 0),  # +x (east)
            (-1, 0, 0), # -x (west)
            (0, 1, 0),  # +y (up)
            (0, -1, 0), # -y (down)
            (0, 0, 1),  # +z (south)
            (0, 0, -1), # -z (north)
        ]
        num_blocks = len(self.blocks)
        face_hash_map = {d: defaultdict(set) for d in directions}
        block_faces = {d: [] for d in directions}
        for idx, block in enumerate(self.blocks):
            for d in directions:
                face, _ = self._get_block_faces_for_connection(block, block, d)
                face_hash = hash(face.tobytes())
                face_hash_map[d][face_hash].add(idx)
                block_faces[d].append(face)
        neighbor_iter = (
            tqdm(range(num_blocks), desc='Inferring connections', file=sys.stdout)
            if _TQDM_AVAILABLE else range(num_blocks)
        )
        args_list = [
            (
                idx1,
                block_faces,
                face_hash_map,
                directions,
                self.blocks,
                self.block_origins,
                self.neighbor_distance,
                self.similarity_threshold,
                self.material_compatibility_map
            ) for idx1 in range(num_blocks)
        ]
        results = []
        if _TQDM_AVAILABLE:
            pbar = tqdm(total=num_blocks, desc='Inferring connections', file=sys.stdout)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(check_connections_for_block, args_list, chunksize=8):
                idx1, local_allowed = res
                for d, idx2s in local_allowed.items():
                    self.allowed_neighbors[idx1][d].update(idx2s)
                if _TQDM_AVAILABLE:
                    pbar.update(1)
        if _TQDM_AVAILABLE:
            pbar.close()
        else:
            print("Finished inferring connections.")

    def _get_sample_neighbor_region(self, block_origin, direction, n):
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
            min_shape = np.minimum(arr1.shape, arr2.shape)
            arr1 = arr1[tuple(slice(0, s) for s in min_shape)]
            arr2 = arr2[tuple(slice(0, s) for s in min_shape)]
        mat1 = arr1[..., 3].reshape(-1)
        mat2 = arr2[..., 3].reshape(-1)
        color1 = arr1[..., :3].reshape(-1, 3)
        color2 = arr2[..., :3].reshape(-1, 3)
        if mat1.size == 0 or mat2.size == 0:
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
            region1_from_sample = self._get_sample_neighbor_region(block1_origin, direction, n)

            if region1_from_sample.size > 0:
                arr1_to_compare = region1_from_sample
                arr2_to_compare = self._get_block_region_for_connection(block2, direction, n)
                attempt_region_comparison = True

        if not attempt_region_comparison:
            arr1_to_compare, arr2_to_compare = self._get_block_faces_for_connection(block1, block2, direction)

        return self._compare_faces_or_regions(arr1_to_compare, arr2_to_compare)

    def _hash_block(self, block: np.ndarray) -> int:
        return hash(block.tobytes())

    def get_blocks(self) -> List[np.ndarray]:
        return self.blocks

    def get_allowed_neighbors(self) -> Dict[int, Dict[Tuple[int, int, int], List[int]]]:
        return {idx: {d: list(neighs) for d, neighs in dirs.items()} for idx, dirs in self.allowed_neighbors.items()}

    def get_block_objects(self):
        """
        Returns a list of Block objects (from wfc.py) with unique names and allowed_neighbors using block names.
        Adds can_be_ground metadata if the block origin y==0.
        """
        idx_to_name = {i: f"block_{i}" for i in range(len(self.blocks))}
        allowed_neighbors = self.get_allowed_neighbors()
        block_objects = []
        for idx, block_data in enumerate(self.blocks):
            name = idx_to_name[idx]
            neighbors = allowed_neighbors.get(idx, {})
            neighbors_named = {
                direction: [idx_to_name[n_idx] for n_idx in neighbor_indices]
                for direction, neighbor_indices in neighbors.items()
            }
            origin = self.block_origins[idx]
            metadata = {}
            if origin[1] == 0:
                metadata['can_be_ground'] = True
            else:
                metadata['can_be_ground'] = False
            weight = self.block_count_by_index[idx] / len(self.blocks) if not self.allow_repeated_blocks else 1.0
            block_objects.append(Block(name, block_data, allowed_neighbors=neighbors_named, metadata=metadata, weight=weight))
        return block_objects

    def save_block_objects(self, filename):
        """
        Save the block objects (as dicts) to a numpy file for later loading.
        If a file with the same name exists, append a number to the filename.
        """
        import os
        base, ext = os.path.splitext(filename)
        counter = 1
        new_filename = filename
        while os.path.exists(new_filename):
            new_filename = f"{base}_{counter}{ext}"
            counter += 1
        block_objects = self.get_block_objects()
        block_dicts = []
        for block in block_objects:
            block_dicts.append({
            'name': block.name,
            'block_data': block.data.tolist(),  # convert numpy array to list for JSON serialization
            'allowed_neighbors': block.allowed_neighbors,
            'metadata': block.metadata,
            'weight': block.weight
            })
        with open(new_filename, 'w') as f:
            json.dump(block_dicts, f)
        print(f"Saved {len(block_dicts)} block objects to {new_filename}")

    @staticmethod
    def from_saved_scene(filename, block_shape, similarity_threshold=0.95, neighbor_distance=0, material_compatibility_map=None, allow_repeated_blocks=False):
        """
        Load a sample scene from a saved ndarray file and create a SampleBlockExtractor.
        """
        sample_scene = np.load(filename)
        return SampleBlockExtractor(
            sample_scene,
            block_shape,
            similarity_threshold=similarity_threshold,
            neighbor_distance=neighbor_distance,
            material_compatibility_map=material_compatibility_map,
            allow_repeated_blocks=allow_repeated_blocks,
        )


def load_block_objects(filename):
    """
    Load block objects from a JSON or NPY file saved by save_block_objects.
    Returns a list of Block objects.
    """
    ext = os.path.splitext(filename)[1].lower()
    block_objects = []
    if ext == ".json":
        with open(filename, 'r') as f:
            block_dicts = json.load(f)
    elif ext == ".npy":
        block_dicts = np.load(filename, allow_pickle=True)
        if hasattr(block_dicts, 'tolist'):
            block_dicts = block_dicts.tolist()
    else:
        raise ValueError("Unsupported file extension: {}".format(ext))
    for block_dict in block_dicts:
        name = block_dict['name']
        data = np.array(block_dict['block_data'])
        allowed_neighbors = block_dict['allowed_neighbors']
        metadata = block_dict.get('metadata', {})
        weight = block_dict.get('weight', 1.0)
        block_objects.append(Block(name, data, allowed_neighbors=allowed_neighbors, metadata=metadata, weight=weight))
    return block_objects




