from __future__ import annotations

import cv2
import numpy as np

from .config import GuidanceConfig
from .geometry import camera_points_to_ground, ground_to_camera
from .pipeline import GuidanceOutput
from .types import Frame, RowPerception


def _hint_color(hint: str):
    if hint == "CENTER":
        return (60, 220, 90)
    if hint == "HOLD":
        return (90, 90, 90)
    return (80, 180, 255)


def _silhouette(frame: Frame, perc: RowPerception, config: GuidanceConfig) -> np.ndarray:
    depth = frame.depth_m
    h, w = depth.shape
    sil = np.zeros((h, w, 3), dtype=np.uint8)
    z0 = config.desk_depth_min_m if config.desk_mode else config.depth_min_m
    z1 = config.desk_depth_max_m if config.desk_mode else config.depth_max_m
    valid = (depth > z0) & (depth < z1)
    if not np.any(valid):
        return sil
    vv, uu = np.indices(depth.shape)
    z = depth[valid]
    x, y_down, z = frame.intrinsics.pixel_to_cam(uu[valid].astype(np.float32), vv[valid].astype(np.float32), z)
    if config.desk_mode:
        keep = np.abs(y_down) <= config.desk_y_band_m
        xs = x[keep]
        us = uu[valid][keep]
        vs = vv[valid][keep]
    else:
        gx, agl, _gz = camera_points_to_ground(x, y_down, z, config.camera_pitch_rad, config.camera_height_m)
        keep = (agl >= config.trunk_height_min_m) & (agl <= config.trunk_height_max_m)
        xs = gx[keep]
        us = uu[valid][keep]
        vs = vv[valid][keep]
    left_ok = perc.left_line is not None
    right_ok = perc.right_line is not None
    for u, v, xv in zip(us, vs, xs):
        if xv < -0.02:
            sil[int(v), int(u)] = (230, 220, 90) if left_ok else (90, 90, 90)
        elif xv > 0.02:
            sil[int(v), int(u)] = (70, 165, 255) if right_ok else (90, 90, 90)
        else:
            sil[int(v), int(u)] = (70, 70, 70)
    sil = cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cx = int(round(frame.intrinsics.cx))
    cv2.line(sil, (cx, 0), (cx, h - 1), (50, 50, 50), 1)
    for t in perc.trunks:
        if t.u is not None and t.v is not None:
            p = (int(round(t.u)), int(round(t.v)))
        elif config.desk_mode:
            u, v = frame.intrinsics.cam_to_pixel(
                np.array([t.x_m], np.float32), np.array([0.0], np.float32), np.array([t.z_m], np.float32)
            )
            if not np.isfinite(u[0]):
                continue
            p = (int(round(u[0])), int(round(v[0])))
        else:
            cx3, cy3, cz3 = ground_to_camera(
                np.array([t.x_m], np.float32), np.array([0.8], np.float32), np.array([t.z_m], np.float32),
                config.camera_pitch_rad, config.camera_height_m,
            )
            u, v = frame.intrinsics.cam_to_pixel(cx3, cy3, cz3)
            if not np.isfinite(u[0]):
                continue
            p = (int(round(u[0])), int(round(v[0])))
        ring = (230, 220, 90) if t.side == "left" else (70, 165, 255)
        cv2.circle(sil, p, 16, ring, 2, cv2.LINE_AA)
    return sil


def annotate(frame: Frame, perc: RowPerception, out: GuidanceOutput, config: GuidanceConfig):
    sil = _silhouette(frame, perc, config)
    h, w = sil.shape[:2]
    banner_h = 150
    canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
    canvas[:] = (8, 8, 8)
    canvas[banner_h:] = sil
    color = _hint_color(out.hint)
    cv2.rectangle(canvas, (0, 0), (w, banner_h), (12, 12, 12), -1)
    cv2.rectangle(canvas, (0, banner_h - 4), (w, banner_h), color, -1)
    title = "<  LEFT" if out.hint == "LEFT" else ("RIGHT  >" if out.hint == "RIGHT" else out.hint)
    ts, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)
    cv2.putText(canvas, title, ((w - ts[0]) // 2, 78), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5, cv2.LINE_AA)
    half, cell_w, cell_h, gap = 5, 28, 22, 6
    bar_w = 11 * cell_w + 10 * gap
    x0, y0 = (w - bar_w) // 2, 104
    for i in range(-half, half + 1):
        x = x0 + (i + half) * (cell_w + gap)
        on = i == 0 and out.hint == "CENTER"
        if out.lightbar > 0 and i < 0 and i >= -out.lightbar:
            on = True
        if out.lightbar < 0 and i > 0 and i <= -out.lightbar:
            on = True
        fill = (60, 220, 90) if on and i == 0 else (color if on else (28, 28, 28))
        cv2.rectangle(canvas, (x, y0), (x + cell_w, y0 + cell_h), fill, -1)
        cv2.rectangle(canvas, (x, y0), (x + cell_w, y0 + cell_h), (50, 50, 50), 1)
    lat = "--" if out.lateral_error_m is None else f"{out.lateral_error_m:+.2f} m"
    line = f"lat {lat}   trunks {out.trunks}   conf {out.confidence:.2f}"
    cv2.putText(canvas, line, (16, banner_h + h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas
