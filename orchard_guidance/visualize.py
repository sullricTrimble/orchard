from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import GuidanceConfig
from .geometry import ground_to_camera
from .pipeline import GuidanceOutput
from .types import Frame, LineXZ, RowPerception


def _project_ground(x, z, height, frame, config):
    cx, cy, cz = ground_to_camera(
        np.array([x], np.float32), np.array([height], np.float32), np.array([z], np.float32),
        config.camera_pitch_rad, config.camera_height_m,
    )
    if cz[0] < 0.4:
        return None
    u, v = frame.intrinsics.cam_to_pixel(cx, cy, cz)
    if not np.isfinite(u[0]) or not np.isfinite(v[0]):
        return None
    return int(round(u[0])), int(round(v[0]))


def _draw_line_xz(img, line, frame, config, color, height=0.9):
    pts = []
    for z in np.linspace(1.6, 18.0, 12):
        p = _project_ground(line.x_at(float(z)), float(z), height, frame, config)
        if p is not None:
            pts.append(p)
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, 2, cv2.LINE_AA)


def _depth_color(depth, max_m=16.0):
    vis = np.clip(depth / max_m, 0, 1)
    color = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[depth <= 0] = 0
    return color


def _birdseye(perc, config, h=480, w=280):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (18, 22, 20)
    z_max, x_max = 14.0, 4.0

    def to_pix(x, z):
        return int((x / x_max * 0.5 + 0.5) * (w - 1)), int((1.0 - z / z_max) * (h - 1))

    cv2.line(img, to_pix(0, 0), to_pix(0, z_max), (50, 60, 50), 1)
    for z in (4, 8, 12):
        cv2.line(img, to_pix(-x_max, z), to_pix(x_max, z), (40, 45, 40), 1)
        cv2.putText(img, f"{z}m", to_pix(-x_max + 0.15, z + 0.2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 100, 90), 1)

    def polyl(line, color):
        pts = [to_pix(line.x_at(z), z) for z in np.linspace(1.0, z_max, 10)]
        cv2.polylines(img, [np.array(pts, np.int32)], False, color, 2)

    if perc.left_line:
        polyl(perc.left_line, (80, 180, 255))
    if perc.right_line:
        polyl(perc.right_line, (80, 180, 255))
    if perc.centerline:
        polyl(perc.centerline, (60, 220, 80))
    for t in perc.trunks:
        cv2.circle(img, to_pix(t.x_m, t.z_m), 6, (40, 90, 200) if t.side == "left" else (200, 140, 40), -1)
    cv2.circle(img, to_pix(0.0, 0.2), 7, (0, 255, 255), -1)
    cv2.putText(img, "BEV  (camera at bottom)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 180), 1)
    return img


def annotate(frame: Frame, perc: RowPerception, out: GuidanceOutput, config: GuidanceConfig):
    rgb = frame.rgb.copy()
    if perc.left_line:
        _draw_line_xz(rgb, perc.left_line, frame, config, (80, 180, 255))
    if perc.right_line:
        _draw_line_xz(rgb, perc.right_line, frame, config, (80, 180, 255))
    if perc.centerline:
        _draw_line_xz(rgb, perc.centerline, frame, config, (60, 220, 90), height=0.2)
    if perc.vanishing_point_uv is not None:
        u, v = int(perc.vanishing_point_uv[0]), int(perc.vanishing_point_uv[1])
        cv2.drawMarker(rgb, (u, v), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.circle(rgb, (u, v), 10, (0, 255, 255), 2)
    cx = int(frame.intrinsics.cx)
    cv2.line(rgb, (cx, 0), (cx, rgb.shape[0] - 1), (255, 255, 255), 1)
    for t in perc.trunks:
        p = _project_ground(t.x_m, t.z_m, 0.8, frame, config)
        if p is not None:
            cv2.circle(rgb, p, 7, (0, 80, 255), 2)
    color = {"LEFT": (80, 180, 255), "RIGHT": (80, 180, 255), "CENTER": (60, 220, 90), "HOLD": (80, 80, 80)}.get(out.hint, (200, 200, 200))
    cv2.rectangle(rgb, (12, 12), (360, 118), (0, 0, 0), -1)
    cv2.putText(rgb, f"STEER {out.hint}", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    lat = "—" if out.lateral_error_m is None else f"{out.lateral_error_m:+.2f} m"
    hdg = "—" if out.heading_error_deg is None else f"{out.heading_error_deg:+.1f} deg"
    spd = "—" if out.speed_mps is None else f"{out.speed_mps:.2f} m/s"
    cv2.putText(rgb, f"lat {lat}   yaw {hdg}", (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(rgb, f"speed {spd}   trunks {out.trunks}", (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    bev = _birdseye(perc, config, h=rgb.shape[0], w=280)
    depth = cv2.resize(_depth_color(frame.depth_m), (280, rgb.shape[0] // 2))
    combo_r = np.vstack([bev[: rgb.shape[0] // 2], depth])
    if combo_r.shape[0] != rgb.shape[0]:
        combo_r = cv2.resize(combo_r, (280, rgb.shape[0]))
    return np.hstack([rgb, combo_r])
