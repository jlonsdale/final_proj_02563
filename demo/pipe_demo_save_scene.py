import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from wfc import WaveFunctionCollapse3D
from pipe_demo import wfc

# Save the WFC scene as a numpy ndarray file
save_path = os.path.join(os.path.dirname(__file__), 'pipe_demo_scene.npy')
wfc.save_scene_as_ndarray(save_path)
print(f"WFC scene saved to {save_path}")
