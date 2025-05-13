import taichi as ti
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

from sample_block_extractor import *
from wfc import WaveFunctionCollapse3D
from scene import Scene
from sample_from_scene_demo import *
if __name__ == '__main__':
    # Load block objects from blocks.json.np
    block_objects = load_block_objects('sample_sceen_air_math_light.json')
    print(f"Loaded {len(block_objects)} unique blocks from blocks.json.npy.")

    # Optionally, load a sample scene for visualization (if needed)
    sample_scene = np.load("example_castle_scene_3.npy")
    sample_scene = make_sample_scene()
    print(f"Sample scene shape: {sample_scene.shape}")
    wfc_size = (2,2,2)
    # Create a WFC instance
    wfc = WaveFunctionCollapse3D(*wfc_size, block_objects, seed=42, enforce_ground_constraint=True)

    # Create and configure the scene
    scene = Scene(voxel_edges=0.1, exposure=1)
    scene.set_floor(0, (1.0, 1.0, 1.0))
    scene.set_background_color((0.5, 0.5, 0.4))
    scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6))

    # Visualize blocks and their neighbors at z=30
 
    block_debugger_and_viewer_in_scene(
        scene,
        sample_scene,
        block_objects,
        base_z=(wfc_size[2] * block_objects[0].data.shape[2]) + 20,)

    # Run WFC and build the scene
    wfc.collapse()
    wfc.build_scene(scene)
    scene.finish()
