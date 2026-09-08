import numpy as np

from orchard_guidance.geometry import camera_points_to_ground, ground_to_camera, rotation_world_to_camera


def test_roundtrip_ground_camera():
    pitch = np.deg2rad(-6.0)
    h = 1.55
    x = np.array([-2.25, 2.25, 0.0], dtype=np.float32)
    agl = np.array([0.8, 0.8, 0.0], dtype=np.float32)
    z = np.array([6.0, 6.0, 8.0], dtype=np.float32)
    cx, cy, cz = ground_to_camera(x, agl, z, pitch, h)
    gx, gagl, gz = camera_points_to_ground(cx, cy, cz, pitch, h)
    assert np.allclose(gx, x, atol=1e-3)
    assert np.allclose(gagl, agl, atol=1e-3)
    assert np.allclose(gz, z, atol=1e-3)


def test_look_down_puts_ground_in_lower_image():
    r = rotation_world_to_camera(0.0, np.deg2rad(-6.0))
    p_cam = r @ np.array([0.0, -1.55, 8.0], dtype=np.float32)
    assert p_cam[2] > 0
    assert p_cam[1] > 0
