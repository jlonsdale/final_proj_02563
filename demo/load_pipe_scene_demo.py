import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
import numpy as np
from sample_block_extractor import SampleBlockExtractor
from scene import Scene
from wfc import build_kernel

# Path to the saved WFC scene
scene_path = os.path.join(os.path.dirname(__file__), 'pipe_demo_scene.npy')

# Load the saved scene ndarray
sample_scene_blocks = np.load(scene_path)  # shape: (width, height, depth, bx, by, bz, 4)

# Flatten the block grid into a single voxel grid
width, height, depth, bx, by, bz, channels = sample_scene_blocks.shape
sample_scene = sample_scene_blocks.transpose(0,3,1,4,2,5,6).reshape(width*bx, height*by, depth*bz, channels)

# Define block shape for extraction (should match the block size used in pipe_demo)
block_shape = (3, 3, 3)

# Create a SampleBlockExtractor from the saved scene
extractor = SampleBlockExtractor(sample_scene, block_shape)
block_objects = extractor.get_block_objects()
print(f"Extracted {len(block_objects)} unique blocks from saved pipe demo scene.")

# Visualize the first extracted block in a new scene
scene = Scene(voxel_edges=0.1, exposure=1)
scene.set_floor(0, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6))

for i, block in enumerate(block_objects[:8]):
    block.build(scene, (i * (block_shape[0] + 1), 0, 0))

scene.finish()

# Run WFC on the extracted blocks and visualize the result
from wfc import WaveFunctionCollapse3D

wfc2 = WaveFunctionCollapse3D(6, 6, 2, block_objects)
wfc2.collapse()
scene2 = Scene(voxel_edges=0.1, exposure=1)
scene2.set_floor(0, (1.0, 1.0, 1.0))
scene2.set_background_color((0.5, 0.5, 0.4))
scene2.set_directional_light((1, 1, -1), 0.1, (1, 0.8, 0.6))
wfc2.build_scene(scene2)
scene2.finish()
