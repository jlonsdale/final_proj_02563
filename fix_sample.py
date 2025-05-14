import numpy as np

scean = np.load("example_castle_scene.npy")
print(scean.shape)

# cut 7 block from the bottom 
scean = scean[7:, 7:, 7:-7, :]
np.save("example_castle_scene_3.npy", scean)
