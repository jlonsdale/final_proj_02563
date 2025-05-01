use python ver 3.10.0
pip install taichi

Done:

- Use pipes for to test basic functionality for 3d wfc
- Generate a sample tileset from a small voxel scene

ToDo:

- Tweak ruleset for generated tileset:
  - Should we discard the compersion if two array sizes do not match ?
  - Sould we match 100% on materials ?
  - Decide what to do on the edge of the sampled scene - Match on air, itsself, anything or nothing or even a combination ?
  - Work on a simple city sample scene to test the above 
  - Add functionality for pre-collapsing one or more cells ?
 

