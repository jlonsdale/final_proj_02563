import time
import os
from datetime import datetime
import numpy as np
import taichi as ti
from renderer import Renderer
from math_utils import np_normalize, np_rotate_matrix
import __main__


VOXEL_DX = 1 / 64
SCREEN_RES = (1280, 720)
TARGET_FPS = 30
UP_DIR = (0, 1, 0)
HELP_MSG = '''
====================================================
Camera:
* Drag with your left mouse button to rotate
* Press W/A/S/D/Q/E to move
====================================================
'''

MAT_LAMBERTIAN = 1
MAT_LIGHT = 2

class Camera:
    def __init__(self, window, up):
        self._window = window
        self._camera_pos = np.array((0.4, 0.5, 2.0))
        self._lookat_pos = np.array((0.0, 0.0, 0.0))
        self._up = np_normalize(np.array(up))
        self._last_mouse_pos = None

    @property
    def mouse_exclusive_owner(self):
        return True

    def update_camera(self):
        res = self._update_by_wasd()
        res = self._update_by_mouse() or res
        return res

    def _update_by_mouse(self):
        win = self._window
        if not self.mouse_exclusive_owner or not win.is_pressed(ti.ui.LMB):
            self._last_mouse_pos = None
            return False
        mouse_pos = np.array(win.get_cursor_pos())
        if self._last_mouse_pos is None:
            self._last_mouse_pos = mouse_pos
            return False
        # Makes camera rotation feels right
        dx, dy = self._last_mouse_pos - mouse_pos
        self._last_mouse_pos = mouse_pos

        out_dir = self._lookat_pos - self._camera_pos
        leftdir = self._compute_left_dir(np_normalize(out_dir))

        scale = 3
        rotx = np_rotate_matrix(self._up, dx * scale)
        roty = np_rotate_matrix(leftdir, dy * scale)

        out_dir_homo = np.array(list(out_dir) + [0.0])
        new_out_dir = np.matmul(np.matmul(roty, rotx), out_dir_homo)[:3]
        self._lookat_pos = self._camera_pos + new_out_dir

        return True

    def _update_by_wasd(self):
        win = self._window
        tgtdir = self.target_dir
        leftdir = self._compute_left_dir(tgtdir)
        lut = [
            ('w', tgtdir),
            ('a', leftdir),
            ('s', -tgtdir),
            ('d', -leftdir),
            ('e', [0, -1, 0]),
            ('q', [0, 1, 0]),
        ]
        dir = np.array([0.0, 0.0, 0.0])
        pressed = False
        for key, d in lut:
            if win.is_pressed(key):
                pressed = True
                dir += np.array(d)
        if not pressed:
            return False
        dir *= 0.05
        self._lookat_pos += dir
        self._camera_pos += dir
        return True

    @property
    def position(self):
        return self._camera_pos

    @property
    def look_at(self):
        return self._lookat_pos

    @property
    def target_dir(self):
        return np_normalize(self.look_at - self.position)

    def _compute_left_dir(self, tgtdir):
        cos = np.dot(self._up, tgtdir)
        if abs(cos) > 0.999:
            return np.array([-1.0, 0.0, 0.0])
        return np.cross(self._up, tgtdir)


@ti.data_oriented
class Scene:
    def __init__(self, voxel_edges=0.06, exposure=3):
        ti.init(arch=ti.vulkan)
        print(HELP_MSG)
        self.window = ti.ui.Window("Taichi Voxel Renderer",
                                   SCREEN_RES,
                                   vsync=True)
        self.camera = Camera(self.window, up=UP_DIR)
        self.renderer = Renderer(dx=VOXEL_DX,
                                 image_res=SCREEN_RES,
                                 up=UP_DIR,
                                 voxel_edges=voxel_edges,
                                 exposure=exposure)

        self.renderer.set_camera_pos(*self.camera.position)
        if not os.path.exists('screenshot'):
            os.makedirs('screenshot')

    @staticmethod
    @ti.func
    def round_idx(idx_):
        idx = ti.cast(idx_, ti.f32)
        return ti.Vector(
            [ti.round(idx[0]),
             ti.round(idx[1]),
             ti.round(idx[2])]).cast(ti.i32)

    @ti.func
    def set_voxel(self, idx, mat, color):
        self.renderer.set_voxel(self.round_idx(idx), mat, color)

    @ti.func
    def get_voxel(self, idx):
        mat, color = self.renderer.get_voxel(self.round_idx(idx))
        return mat, color

    @ti.kernel
    def _copy_data_to_numpy_kernel(self,
                                   output_array: ti.types.ndarray(),  # type: ignore
                                   min_b: ti.types.vector(3, ti.i32),
                                   max_b: ti.types.vector(3, ti.i32)):
        # Iterate over world coordinates defined by min_b and max_b
        for i, j, k in ti.ndrange((min_b[0], max_b[0] + 1),
                                   (min_b[1], max_b[1] + 1),
                                   (min_b[2], max_b[2] + 1)):
            # Array indices (relative to min_b)
            arr_i = i - min_b[0]
            arr_j = j - min_b[1]
            arr_k = k - min_b[2]

            mat, color_vec = self.get_voxel(ti.Vector([i, j, k]))

            output_array[arr_i, arr_j, arr_k, 0] = color_vec[0]
            output_array[arr_i, arr_j, arr_k, 1] = color_vec[1]
            output_array[arr_i, arr_j, arr_k, 2] = color_vec[2]
            output_array[arr_i, arr_j, arr_k, 3] = ti.cast(mat, ti.f32)

    def save_scene_to_numpy(self, filepath: str):
        """
        Saves the current scene's voxel data to a NumPy .npy file.
        The data is stored as an array of shape (dim_x, dim_y, dim_z, 4),
        where the last dimension contains (r, g, b, material_id).
        """
        if hasattr(self.renderer, 'recompute_bbox') and callable(self.renderer.recompute_bbox):
            self.renderer.recompute_bbox()
        else:
            print("Warning: Renderer does not have a callable recompute_bbox method. Bounding box might not be up-to-date.")

        if not hasattr(self.renderer, 'bbox'):
            print("Error: Renderer does not have bbox attribute. Cannot determine scene bounds to save.")
            return

        # Retrieve Taichi field values into Python-scope Taichi Vector variables
        world_min_bbox_py_vec = self.renderer.bbox[0]  # Already a ti.Vector in Python scope
        world_max_bbox_py_vec = self.renderer.bbox[1]  # Already a ti.Vector in Python scope
        inv_dx_py_float = self.renderer.voxel_inv_dx  # This is a Python float

        # Perform calculations: ti.math.floor on a ti.Vector returns a new ti.Vector.
        # These operations are on Python-scope ti.Vector objects.
        kernel_min_b_fvec = ti.math.floor(world_min_bbox_py_vec * inv_dx_py_float)
        kernel_max_b_fvec = ti.math.floor(world_max_bbox_py_vec * inv_dx_py_float) - 2 # Element-wise subtraction

        # Instead of .cast(ti.i32), convert to Python int lists for kernel
        kernel_min_b_for_kernel = [int(kernel_min_b_fvec[i]) for i in range(3)]
        kernel_max_b_for_kernel = [int(kernel_max_b_fvec[i]) for i in range(3)]

        # For Python-scope dimension calculations and printing, use these lists
        dim_x = kernel_max_b_for_kernel[0] - kernel_min_b_for_kernel[0] + 1
        dim_y = kernel_max_b_for_kernel[1] - kernel_min_b_for_kernel[1] + 1
        dim_z = kernel_max_b_for_kernel[2] - kernel_min_b_for_kernel[2] + 1

        if dim_x <= 0 or dim_y <= 0 or dim_z <= 0:
            world_min_py_list = [float(world_min_bbox_py_vec[i]) for i in range(3)]
            world_max_py_list = [float(world_max_bbox_py_vec[i]) for i in range(3)]
            kernel_min_py_list = kernel_min_b_for_kernel
            kernel_max_py_list = kernel_max_b_for_kernel
            print(f"Warning: Scene bounding box results in non-positive dimensions. "
                  f"World Min: {world_min_py_list}, World Max: {world_max_py_list}. "
                  f"Voxel Index Min: {kernel_min_py_list}, Voxel Index Max: {kernel_max_py_list}. "
                  f"Calculated Dims: ({dim_x}, {dim_y}, {dim_z}). "
                  f"Renderer voxel_dx: {self.renderer.voxel_dx}, inv_dx: {inv_dx_py_float}. "
                  f"Saving an empty array to {filepath}")
            empty_data_array = np.zeros((0, 0, 0, 4), dtype=np.float32)
            np.save(filepath, empty_data_array)
            return

        scene_data_np = np.zeros((dim_x, dim_y, dim_z, 4), dtype=np.float32)

        self._copy_data_to_numpy_kernel(scene_data_np, kernel_min_b_for_kernel, kernel_max_b_for_kernel)

        try:
            np.save(filepath, scene_data_np)
            print(f"Scene data saved to {filepath} with shape {scene_data_np.shape}")
        except Exception as e:
            print(f"Error saving scene to {filepath}: {e}")

    def set_floor(self, height, color):
        self.renderer.floor_height[None] = height
        self.renderer.floor_color[None] = color

    def set_directional_light(self, direction, direction_noise, color):
        self.renderer.set_directional_light(direction, direction_noise, color)

    def set_background_color(self, color):
        self.renderer.background_color[None] = color

    def finish(self):
        self.renderer.recompute_bbox()
        canvas = self.window.get_canvas()
        spp = 1
        while self.window.running:
            should_reset_framebuffer = False

            if self.camera.update_camera():
                self.renderer.set_camera_pos(*self.camera.position)
                look_at = self.camera.look_at
                self.renderer.set_look_at(*look_at)
                should_reset_framebuffer = True

            if should_reset_framebuffer:
                self.renderer.reset_framebuffer()

            t = time.time()
            for _ in range(spp):
                self.renderer.accumulate()
            img = self.renderer.fetch_image()
            if self.window.is_pressed('p'):
                timestamp = datetime.today().strftime('%Y-%m-%d-%H%M%S')
                dirpath = os.getcwd()
                main_filename = os.path.split(__main__.__file__)[1]
                fname = os.path.join(dirpath, 'screenshot', f"{main_filename}-{timestamp}.jpg")
                ti.tools.image.imwrite(img, fname)
                print(f"Screenshot has been saved to {fname}")
            canvas.set_image(img)
            elapsed_time = time.time() - t
            if elapsed_time * TARGET_FPS > 1:
                spp = int(spp / (elapsed_time * TARGET_FPS) - 1)
                spp = max(spp, 1)
            else:
                spp += 1
            self.window.show()
