from __future__ import annotations

import numpy as np


def rotation_world_to_camera(yaw_rad: float, pitch_aero_rad: float) -> np.ndarray:
    """World (X right, Y up, Z along-row) to camera (X right, Y down, Z forward).

    pitch_aero_rad uses aviation sign: negative is nose-down / look-down.
    """
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    r_yaw = np.array([[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]], dtype=np.float32)
    look_down = -pitch_aero_rad
    cp, sp = np.cos(look_down), np.sin(look_down)
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    r_flip = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return r_flip @ r_pitch @ r_yaw


def camera_points_to_ground(x, y_down, z, pitch_aero_rad, camera_height_m):
    r = rotation_world_to_camera(0.0, pitch_aero_rad)
    pts = np.stack([x, y_down, z], axis=-1).astype(np.float32)
    rel = pts @ r
    height_agl = camera_height_m + rel[:, 1]
    return rel[:, 0], height_agl, rel[:, 2]


def ground_to_camera(x_right, height_agl, z_forward, pitch_aero_rad, camera_height_m):
    r = rotation_world_to_camera(0.0, pitch_aero_rad)
    rel = np.stack([x_right, height_agl - camera_height_m, z_forward], axis=-1).astype(np.float32)
    cam = (r @ rel.T).T
    return cam[:, 0], cam[:, 1], cam[:, 2]
