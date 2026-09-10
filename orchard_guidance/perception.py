from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import GuidanceConfig
from .geometry import camera_points_to_ground, ground_to_camera
from .types import Frame, LineXZ, RowPerception, Trunk


def _fit_line_xz(xs: np.ndarray, zs: np.ndarray, thresh: float) -> Optional[LineXZ]:
    xs = np.asarray(xs, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    if xs.size < 2:
        return None
    mask = np.ones(xs.size, dtype=bool)
    intercept = slope = 0.0
    for _ in range(4):
        if int(mask.sum()) < 2:
            return None
        a = np.stack([np.ones(int(mask.sum())), zs[mask]], axis=1)
        coef, *_ = np.linalg.lstsq(a, xs[mask], rcond=None)
        intercept, slope = float(coef[0]), float(coef[1])
        resid = np.abs(intercept + slope * zs - xs)
        mask = resid < thresh
    if int(mask.sum()) < 2:
        return None
    return LineXZ(intercept=intercept, slope=slope)


def _cluster_xz(xs, zs, eps, min_pts):
    n = xs.size
    if n == 0:
        return []
    used = np.zeros(n, dtype=bool)
    trunks = []
    for i in np.argsort(zs):
        if used[i]:
            continue
        dist2 = (xs - xs[i]) ** 2 + (zs - zs[i]) ** 2
        members = dist2 < eps * eps
        count = int(members.sum())
        if count < min_pts:
            continue
        used[members] = True
        trunks.append((float(xs[members].mean()), float(zs[members].mean()), count))
    trunks.sort(key=lambda t: t[1])
    return trunks


def _intersect_image_lines(l0, l1):
    a0, b0, c0 = l0
    a1, b1, c1 = l1
    det = a0 * b1 - a1 * b0
    if abs(det) < 1e-8:
        return None
    x = (b0 * c1 - b1 * c0) / det
    y = (c0 * a1 - c1 * a0) / det
    return float(x), float(y)


def _line_from_two_points(u0, v0, u1, v1):
    return (v0 - v1, u1 - u0, u0 * v1 - u1 * v0)


def _vanishing_from_rgb(rgb, cx):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    h, w = gray.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=40, minLineLength=int(h * 0.18), maxLineGap=25)
    if lines is None:
        return None
    segs = lines.reshape(-1, 4)
    left_pts, right_pts = [], []
    for x1, y1, x2, y2 in segs:
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = x2 - x1, y2 - y1
        if float(np.hypot(dx, dy)) < 30:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle < 18 or angle > 78:
            continue
        mid_x = 0.5 * (x1 + x2)
        if mid_x < cx and x2 < x1:
            left_pts.append((x1, y1, x2, y2))
        elif mid_x > cx and x2 > x1:
            right_pts.append((x1, y1, x2, y2))
        elif mid_x < cx and x2 > x1:
            left_pts.append((x1, y1, x2, y2))
        elif mid_x > cx and x2 < x1:
            right_pts.append((x1, y1, x2, y2))
    if len(left_pts) < 1 or len(right_pts) < 1:
        return None

    def consensus(segs):
        arr = np.array(segs, dtype=np.float64)
        u0, v0 = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        u1, v1 = float(arr[:, 2].mean()), float(arr[:, 3].mean())
        if abs(v1 - v0) < 1:
            return None
        return _line_from_two_points(u0, v0, u1, v1)

    l_line, r_line = consensus(left_pts), consensus(right_pts)
    if l_line is None or r_line is None:
        return None
    return _intersect_image_lines(l_line, r_line)


def _project_line(line, z0, z1, height_agl, frame, config):
    xs = np.array([line.x_at(z0), line.x_at(z1)], dtype=np.float32)
    hs = np.array([height_agl, height_agl], dtype=np.float32)
    zs = np.array([z0, z1], dtype=np.float32)
    cx, cy, cz = ground_to_camera(xs, hs, zs, config.camera_pitch_rad, config.camera_height_m)
    if np.any(cz < 0.3):
        return None
    u, v = frame.intrinsics.cam_to_pixel(cx, cy, cz)
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        return None
    return (float(u[0]), float(v[0])), (float(u[1]), float(v[1]))


def _inner_envelope(xs, zs, side, z_bin=0.9, min_pts=8):
    if xs.size == 0:
        return np.array([]), np.array([])
    z0, z1 = float(zs.min()), float(zs.max())
    bins = np.arange(z0, z1 + z_bin, z_bin)
    ox, oz = [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (zs >= b0) & (zs < b1)
        if int(m.sum()) < min_pts:
            continue
        ox.append(float(xs[m].max() if side == "left" else xs[m].min()))
        oz.append(float(0.5 * (b0 + b1)))
    return np.asarray(ox, dtype=np.float64), np.asarray(oz, dtype=np.float64)


def _plausible_vp(vp, width, height):
    u, v = vp
    return (0.18 * width) < u < (0.82 * width) and (-0.15 * height) < v < (0.62 * height)


def _cluster_pens(xs, ys, zs, eps, min_pts):
    n = xs.size
    if n == 0:
        return []
    used = np.zeros(n, dtype=bool)
    out = []
    for i in np.argsort(zs):
        if used[i]:
            continue
        dist2 = (xs - xs[i]) ** 2 + (zs - zs[i]) ** 2
        members = dist2 < eps * eps
        count = int(members.sum())
        if count < min_pts:
            continue
        used[members] = True
        mx, my, mz = float(xs[members].mean()), float(ys[members].mean()), float(zs[members].mean())
        xz_span = float(np.hypot(xs[members].max() - xs[members].min(), zs[members].max() - zs[members].min()))
        y_span = float(ys[members].max() - ys[members].min())
        out.append((mx, my, mz, count, xz_span, y_span))
    return out


def perceive_pens(frame: Frame, config: GuidanceConfig) -> RowPerception:
    """Lock onto two compact nearby objects (pens) and steer between them."""
    result = RowPerception()
    depth = frame.depth_m
    k = frame.intrinsics
    z0, z1 = config.desk_depth_min_m, config.desk_depth_max_m
    valid = (depth > z0) & (depth < z1)
    if int(valid.sum()) < 40:
        result.notes.append("too_few_depth_pixels")
        return result

    vv, uu = np.indices(depth.shape)
    z = depth[valid]
    x, y_down, z = k.pixel_to_cam(uu[valid].astype(np.float32), vv[valid].astype(np.float32), z)
    band = np.abs(y_down) <= config.desk_y_band_m
    x, y_down, z = x[band], y_down[band], z[band]
    if x.size < 20:
        result.notes.append("no_band")
        return result

    clusters = []
    for mx, my, mz, n, xz_span, y_span in _cluster_pens(
        x, y_down, z, config.desk_cluster_eps_m, config.desk_cluster_min_points
    ):
        if xz_span > config.desk_max_span_m:
            continue
        if y_span < config.desk_min_vertical_m and n < config.desk_cluster_min_points * 3:
            continue
        clusters.append((mx, my, mz, n, xz_span, y_span))

    left = [c for c in clusters if c[0] < -0.02]
    right = [c for c in clusters if c[0] > 0.02]
    left.sort(key=lambda c: c[2])
    right.sort(key=lambda c: c[2])
    if not left:
        result.notes.append("left_missing")
    if not right:
        result.notes.append("right_missing")
    if not left or not right:
        result.notes.append("need_both")
        return result

    lx, ly, lz, ln, lspan, _lyspan = left[0]
    rx, ry, rz, rn, rspan, _ryspan = right[0]
    gap = rx - lx
    if gap < config.desk_min_gap_m or gap > config.desk_max_gap_m:
        result.notes.append("gap_rejected")
        return result

    center_x = 0.5 * (lx + rx)
    lat = -center_x
    if abs(lat) > config.desk_max_lat_m:
        result.notes.append("lat_rejected")
        return result

    result.left_line = LineXZ(intercept=lx, slope=0.0)
    result.right_line = LineXZ(intercept=rx, slope=0.0)
    result.centerline = LineXZ(intercept=center_x, slope=0.0)
    result.lateral_error_m = float(lat)
    result.heading_error_rad = 0.0
    result.row_width_m = float(gap)
    result.depth_range_m = float(max(lz, rz))
    lu, lv = k.cam_to_pixel(np.array([lx]), np.array([ly]), np.array([lz]))
    ru, rv = k.cam_to_pixel(np.array([rx]), np.array([ry]), np.array([rz]))
    result.trunks = [
        Trunk(side="left", x_m=float(lx), z_m=float(lz), u=float(lu[0]), v=float(lv[0]), n_points=int(ln)),
        Trunk(side="right", x_m=float(rx), z_m=float(rz), u=float(ru[0]), v=float(rv[0]), n_points=int(rn)),
    ]
    result.confidence = _pen_confidence(ln, rn, lz, rz, lspan, rspan, gap, config)
    return result


def _pen_confidence(ln, rn, lz, rz, lspan, rspan, gap, config: GuidanceConfig) -> float:
    """How clean the two-pen lock is (not how centered you are)."""

    def unit(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    points = unit(min(ln, rn) / 60.0)
    depth_match = unit(1.0 - abs(lz - rz) / 0.30)
    compact = unit(1.0 - max(lspan, rspan) / max(config.desk_max_span_m, 1e-3))
    gap_q = unit(1.0 - abs(gap - 0.30) / 0.40)
    return float(np.clip(0.30 + 0.28 * points + 0.22 * depth_match + 0.12 * compact + 0.08 * gap_q, 0.05, 0.99))


def perceive(frame: Frame, config: GuidanceConfig) -> RowPerception:
    if config.desk_mode:
        return perceive_pens(frame, config)
    result = RowPerception()
    depth = frame.depth_m
    k = frame.intrinsics
    valid = (depth > config.depth_min_m) & (depth < config.depth_max_m)
    if int(valid.sum()) < 80:
        result.notes.append("too_few_depth_pixels")
        return result

    vv, uu = np.indices(depth.shape)
    z = depth[valid]
    x, y_down, z = k.pixel_to_cam(uu[valid].astype(np.float32), vv[valid].astype(np.float32), z)
    gx, agl, gz = camera_points_to_ground(x, y_down, z, config.camera_pitch_rad, config.camera_height_m)
    band = (
        (agl >= config.trunk_height_min_m)
        & (agl <= config.trunk_height_max_m)
        & (gz >= config.depth_min_m)
        & (gz <= config.depth_max_m)
    )
    gx, gz = gx[band], gz[band]
    if gx.size < 40:
        result.notes.append("no_trunk_band")

    left_mask = gx < -config.center_keepout_m
    right_mask = gx > config.center_keepout_m
    left_env_x, left_env_z = _inner_envelope(gx[left_mask], gz[left_mask], "left")
    right_env_x, right_env_z = _inner_envelope(gx[right_mask], gz[right_mask], "right")

    if left_env_x.size >= config.min_trunks_for_line:
        result.left_line = _fit_line_xz(left_env_x, left_env_z, max(config.ransac_thresh_m, 0.55))
    elif int(left_mask.sum()) > 30:
        result.left_line = _fit_line_xz(gx[left_mask], gz[left_mask], config.ransac_thresh_m)
        result.notes.append("left_raw_points")

    if right_env_x.size >= config.min_trunks_for_line:
        result.right_line = _fit_line_xz(right_env_x, right_env_z, max(config.ransac_thresh_m, 0.55))
    elif int(right_mask.sum()) > 30:
        result.right_line = _fit_line_xz(gx[right_mask], gz[right_mask], config.ransac_thresh_m)
        result.notes.append("right_raw_points")

    def trunks_along(line, xs, zs, side):
        if xs.size == 0:
            return
        near = np.abs(line.intercept + line.slope * zs - xs) < 0.50
        for tx, tz, n in _cluster_xz(xs[near], zs[near], config.cluster_eps_m, config.cluster_min_points):
            result.trunks.append(Trunk(side=side, x_m=tx, z_m=tz, n_points=n))

    if result.left_line is not None:
        trunks_along(result.left_line, gx[left_mask], gz[left_mask], "left")
    if result.right_line is not None:
        trunks_along(result.right_line, gx[right_mask], gz[right_mask], "right")

    if result.left_line and result.right_line:
        result.centerline = LineXZ(
            intercept=0.5 * (result.left_line.intercept + result.right_line.intercept),
            slope=0.5 * (result.left_line.slope + result.right_line.slope),
        )
        result.row_width_m = abs(result.right_line.x_at(6.0) - result.left_line.x_at(6.0))
        result.lateral_error_m = -result.centerline.intercept
        result.heading_error_rad = -result.centerline.heading_rad()
        if result.trunks:
            result.depth_range_m = float(max(t.z_m for t in result.trunks))
    elif result.left_line:
        assumed = result.left_line.intercept + config.row_width_m
        result.centerline = LineXZ(
            intercept=0.5 * (result.left_line.intercept + assumed), slope=result.left_line.slope
        )
        result.lateral_error_m = -result.centerline.intercept
        result.heading_error_rad = -result.centerline.heading_rad()
        result.notes.append("right_side_missing_assumed_width")
    elif result.right_line:
        assumed = result.right_line.intercept - config.row_width_m
        result.centerline = LineXZ(
            intercept=0.5 * (result.right_line.intercept + assumed), slope=result.right_line.slope
        )
        result.lateral_error_m = -result.centerline.intercept
        result.heading_error_rad = -result.centerline.heading_rad()
        result.notes.append("left_side_missing_assumed_width")

    vp_rgb = _vanishing_from_rgb(frame.rgb, k.cx)
    if vp_rgb is not None and _plausible_vp(vp_rgb, k.width, k.height):
        result.heading_from_rgb_rad = -float(np.arctan((vp_rgb[0] - k.cx) / k.fx))
        result.vanishing_point_uv = vp_rgb

    if result.left_line and result.right_line:
        p_l = _project_line(result.left_line, 2.0, 28.0, 0.9, frame, config)
        p_r = _project_line(result.right_line, 2.0, 28.0, 0.9, frame, config)
        if p_l and p_r:
            vp_depth = _intersect_image_lines(
                _line_from_two_points(*p_l[0], *p_l[1]),
                _line_from_two_points(*p_r[0], *p_r[1]),
            )
            if vp_depth is not None and np.isfinite(vp_depth).all():
                result.vanishing_point_uv = vp_depth
                if (
                    result.heading_error_rad is not None
                    and result.heading_from_rgb_rad is not None
                    and abs(result.heading_error_rad - result.heading_from_rgb_rad) < np.deg2rad(8.0)
                ):
                    result.heading_error_rad = 0.70 * result.heading_error_rad + 0.30 * result.heading_from_rgb_rad
                    result.notes.append("fused_rgb_heading")

    n_trk = len(result.trunks)
    conf = 0.0
    if result.centerline is not None:
        conf += 0.45
    if n_trk >= 4:
        conf += 0.25
    elif n_trk >= 2:
        conf += 0.12
    if result.left_line and result.right_line:
        conf += 0.20
    if result.vanishing_point_uv is not None:
        conf += 0.10
    result.confidence = float(min(1.0, conf))
    return result
