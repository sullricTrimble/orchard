from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float, vfov_deg: float) -> "Intrinsics":
        hfov = np.deg2rad(hfov_deg)
        vfov = np.deg2rad(vfov_deg)
        fx = width / (2.0 * np.tan(hfov / 2.0))
        fy = height / (2.0 * np.tan(vfov / 2.0))
        return cls(fx=float(fx), fy=float(fy), cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    def pixel_to_cam(self, u: np.ndarray, v: np.ndarray, z: np.ndarray):
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return x, y, z

    def cam_to_pixel(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        z_safe = np.maximum(z, 1e-6)
        u = self.fx * x / z_safe + self.cx
        v = self.fy * y / z_safe + self.cy
        return u, v


@dataclass
class Frame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: Intrinsics
    timestamp_s: float


@dataclass
class LineXZ:
    intercept: float
    slope: float

    def x_at(self, z: float) -> float:
        return self.intercept + self.slope * z

    def heading_rad(self) -> float:
        return float(np.arctan(self.slope))


@dataclass
class Trunk:
    side: str
    x_m: float
    z_m: float
    u: Optional[float] = None
    v: Optional[float] = None
    n_points: int = 0


@dataclass
class RowPerception:
    left_line: Optional[LineXZ] = None
    right_line: Optional[LineXZ] = None
    centerline: Optional[LineXZ] = None
    lateral_error_m: Optional[float] = None
    heading_error_rad: Optional[float] = None
    heading_from_rgb_rad: Optional[float] = None
    vanishing_point_uv: Optional[tuple[float, float]] = None
    trunks: list[Trunk] = field(default_factory=list)
    row_width_m: Optional[float] = None
    depth_range_m: Optional[float] = None
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
