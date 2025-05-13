import taichi as ti
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

from sample_block_extractor import *
from wfc import WaveFunctionCollapse3D
from scene import Scene

if __name__ == '__main__':
    # Load block objects from blocks.json.np
    block_objects = load_block_objects('blocks_23.json')
    print(f"Loaded {len(block_objects)} unique blocks from blocks.json.npy.")

    # Optionally, load a sample scene for visualization (if needed)
    sample_scene = np.load("example_castle_scene.npy")

    # Create a WFC instance
    wfc = WaveFunctionCollapse3D(10, 3, 10, block_objects, seed=42, enforce_ground_constraint=True)

    # Create and configure the scene
    scene = Scene(voxel_edges=0.1, exposure=1)
    scene.set_floor(0, (1.0, 1.0, 1.0))
    scene.set_background_color((0.5, 0.5, 0.4))
    scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6))

    # Visualize blocks and their neighbors at z=30
    from sample_from_scene_demo import block_debugger_and_viewer_in_scene
    block_debugger_and_viewer_in_scene(
        scene,
        sample_scene,
        block_objects,
        base_z=40)

    # Run WFC and build the scene
    wfc.collapse()
    wfc.build_scene(scene)
    scene.finish()
