from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import GuidanceConfig
from .geometry import rotation_world_to_camera
from .types import Frame, Intrinsics


@dataclass
class TractorPose:
    x_m: float = 0.0
    s_m: float = 4.0
    yaw_rad: float = 0.0
    speed_mps: float = 1.8


@dataclass
class OrchardWorld:
    left_trees: np.ndarray
    right_trees: np.ndarray
    row_width_m: float
    tree_spacing_m: float
    end_z_m: float


def build_orchard(config: GuidanceConfig, rng: np.random.Generator) -> OrchardWorld:
    zs = np.arange(2.0, config.sim_row_length_m, config.tree_spacing_m)
    zs = zs + rng.uniform(-config.sim_spacing_jitter_m, config.sim_spacing_jitter_m, size=zs.shape)
    half = config.row_width_m / 2.0
    left_x = -half + rng.uniform(-config.sim_lateral_jitter_m, config.sim_lateral_jitter_m, size=zs.shape)
    right_x = half + rng.uniform(-config.sim_lateral_jitter_m, config.sim_lateral_jitter_m, size=zs.shape)
    return OrchardWorld(
        left_trees=np.stack([left_x, zs], axis=1),
        right_trees=np.stack([right_x, zs], axis=1),
        row_width_m=config.row_width_m,
        tree_spacing_m=config.tree_spacing_m,
        end_z_m=float(zs.max()) if len(zs) else config.sim_row_length_m,
    )


class OrchardSimulator:
    def __init__(self, config: Optional[GuidanceConfig] = None, seed: int = 3):
        self.config = config or GuidanceConfig()
        self.rng = np.random.default_rng(seed)
        self.world = build_orchard(self.config, self.rng)
        self.pose = TractorPose(
            x_m=self.config.sim_start_lateral_m,
            s_m=3.5,
            yaw_rad=np.deg2rad(self.config.sim_start_yaw_deg),
            speed_mps=self.config.sim_speed_mps,
        )
        self.intrinsics = Intrinsics.from_fov(
            self.config.image_width,
            self.config.image_height,
            self.config.depth_hfov_deg,
            self.config.depth_vfov_deg,
        )
        self.timestamp_s = 0.0
        self.auto_drive = True
        self._last_steer = 0.0
        self._dirs_cache = None

    def reset(self, lateral_m: float, yaw_deg: float, s_m: float = 3.5) -> None:
        self.pose = TractorPose(
            x_m=lateral_m, s_m=s_m, yaw_rad=np.deg2rad(yaw_deg), speed_mps=self.config.sim_speed_mps
        )
        self.timestamp_s = 0.0
        self._last_steer = 0.0

    def apply_steer(self, steer: float) -> None:
        self._last_steer = float(np.clip(steer, -1.0, 1.0))

    def step(self, dt: Optional[float] = None) -> None:
        if dt is None:
            dt = 1.0 / self.config.sim_fps
        pose = self.pose
        if self.auto_drive:
            pose.yaw_rad += (-self._last_steer * 0.42) * dt
        pose.x_m += pose.speed_mps * np.sin(pose.yaw_rad) * dt
        pose.s_m += pose.speed_mps * np.cos(pose.yaw_rad) * dt
        self.timestamp_s += dt

    def read(self) -> Frame:
        rgb, depth = self._render()
        return Frame(rgb=rgb, depth_m=depth, intrinsics=self.intrinsics, timestamp_s=self.timestamp_s)

    def close(self) -> None:
        return None

    def _camera_rotation(self) -> np.ndarray:
        return rotation_world_to_camera(self.pose.yaw_rad, self.config.camera_pitch_rad)

    def _world_to_cam(self, pts_xyz: np.ndarray) -> np.ndarray:
        cam_origin = np.array([self.pose.x_m, self.config.camera_height_m, self.pose.s_m], dtype=np.float32)
        rel = pts_xyz - cam_origin
        return (self._camera_rotation() @ rel.T).T

    def _pixel_rays(self) -> np.ndarray:
        if self._dirs_cache is not None:
            return self._dirs_cache
        k = self.intrinsics
        uu, vv = np.meshgrid(np.arange(k.width, dtype=np.float32), np.arange(k.height, dtype=np.float32))
        x = (uu - k.cx) / k.fx
        y = (vv - k.cy) / k.fy
        self._dirs_cache = np.stack([x, y, np.ones_like(x)], axis=-1)
        return self._dirs_cache

    def _render(self):
        cfg = self.config
        h, w = cfg.image_height, cfg.image_width
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)
        for row in range(h):
            t = row / max(h - 1, 1)
            rgb[row, :] = (int(210 - 70 * t), int(175 - 40 * t), int(125 - 10 * t))
        self._draw_ground(rgb, depth)
        self._draw_trees(rgb, depth)
        rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
        return rgb, depth

    def _draw_ground(self, rgb, depth):
        dirs_cam = self._pixel_rays()
        r = self._camera_rotation()
        dirs_world = dirs_cam @ r
        origin = np.array([self.pose.x_m, self.config.camera_height_m, self.pose.s_m], dtype=np.float32)
        dy = dirs_world[:, :, 1]
        valid = dy < -1e-5
        t = np.zeros(dy.shape, dtype=np.float32)
        t[valid] = -origin[1] / dy[valid]
        hit = origin + dirs_world * t[..., None]
        in_front = valid & (t > 0.4) & (t < 80.0)
        wx, wz = hit[:, :, 0], hit[:, :, 2]
        half = self.world.row_width_m / 2.0
        alley = in_front & (np.abs(wx) < half - 0.25)
        under_tree = in_front & (np.abs(wx) >= half - 0.25) & (np.abs(wx) < half + 1.8)
        beyond = in_front & (wz > self.world.end_z_m + 2.0)
        stripe = ((wz / 0.7).astype(np.int32) + (wx / 0.5).astype(np.int32)) & 1
        gx = rgb.copy()
        gx[alley] = np.where(stripe[alley, None] == 0, np.array([58, 145, 92], np.uint8), np.array([48, 125, 78], np.uint8))
        gx[under_tree] = np.array([32, 72, 48], np.uint8)
        gx[beyond] = np.array([90, 150, 170], np.uint8)
        rgb[:] = gx
        depth[in_front] = t[in_front]
        edge = in_front & (np.abs(np.abs(wx) - (half - 0.35)) < 0.07)
        rgb[edge] = (40, 55, 36)

    def _draw_trees(self, rgb, depth):
        trees = np.concatenate([self.world.left_trees, self.world.right_trees], axis=0)
        sides = ["left"] * len(self.world.left_trees) + ["right"] * len(self.world.right_trees)
        cam = self._world_to_cam(np.column_stack([trees[:, 0], np.zeros(len(trees)), trees[:, 1]]).astype(np.float32))
        order = np.argsort(-cam[:, 2])
        radius = self.config.trunk_radius_m
        k = self.intrinsics
        h, w = rgb.shape[:2]
        for idx in order:
            xw, zw = float(trees[idx, 0]), float(trees[idx, 1])
            if cam[idx, 2] < 0.8 or cam[idx, 2] > 55.0:
                continue
            self._blit_canopy(rgb, depth, xw, zw, sides[idx])
            self._blit_trunk(rgb, depth, xw, zw, radius, k, h, w)

    def _blit_trunk(self, rgb, depth, xw, zw, radius, k, h, w, color=(28, 62, 96)):
        ys = np.array([0.05, 0.05, 2.15, 2.15], dtype=np.float32)
        xs = np.array([xw - radius, xw + radius, xw + radius, xw - radius], dtype=np.float32)
        zs = np.full(4, zw, dtype=np.float32)
        cam = self._world_to_cam(np.stack([xs, ys, zs], axis=1))
        if np.any(cam[:, 2] < 0.4):
            return
        u, v = k.cam_to_pixel(cam[:, 0], cam[:, 1], cam[:, 2])
        poly = np.stack([u, v], axis=1).astype(np.int32)
        if not np.isfinite(poly).all():
            return
        overlay = rgb.copy()
        cv2.fillConvexPoly(overlay, poly, color)
        z_mid = float(np.median(cam[:, 2]))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        paint = (mask > 0) & ((depth == 0) | (z_mid < depth))
        rgb[paint] = overlay[paint]
        depth[paint] = z_mid

    def _blit_canopy(self, rgb, depth, xw, zw, side):
        k = self.intrinsics
        h, w = rgb.shape[:2]
        outward = -1.0 if side == "left" else 1.0
        xs = np.array([xw - 0.15, xw + outward * 1.15, xw + outward * 1.15, xw - 0.15], dtype=np.float32)
        ys = np.array([1.55, 1.55, 3.7, 3.7], dtype=np.float32)
        zs = np.array([zw - 0.7, zw - 0.7, zw + 0.7, zw + 0.7], dtype=np.float32)
        cam = self._world_to_cam(np.stack([xs, ys, zs], axis=1))
        if np.any(cam[:, 2] < 0.5):
            return
        u, v = k.cam_to_pixel(cam[:, 0], cam[:, 1], cam[:, 2])
        poly = np.stack([u, v], axis=1).astype(np.int32)
        if not np.isfinite(poly).all():
            return
        color = (28, 110, 46) if (int(zw) % 2 == 0) else (22, 92, 38)
        overlay = rgb.copy()
        cv2.fillConvexPoly(overlay, poly, color)
        z_mid = float(np.median(cam[:, 2]))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        paint = (mask > 0) & ((depth == 0) | (z_mid < depth))
        rgb[paint] = overlay[paint]
        depth[paint] = z_mid
