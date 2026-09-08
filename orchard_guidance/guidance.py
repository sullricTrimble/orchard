from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import GuidanceConfig
from .types import RowPerception


@dataclass
class SteerCommand:
    lateral_error_m: Optional[float]
    heading_error_deg: Optional[float]
    steer: float
    hint: str
    lightbar: int
    lightbar_cells: int = 11
    confidence: float = 0.0
    reason: str = ""


def compute_steer(perception: RowPerception, config: GuidanceConfig) -> SteerCommand:
    lat = perception.lateral_error_m
    hdg = None if perception.heading_error_rad is None else float(np.degrees(perception.heading_error_rad))
    if lat is None and hdg is None:
        return SteerCommand(None, None, 0.0, "HOLD", 0, confidence=0.0, reason="no_row")

    lat_term = 0.0 if lat is None else config.kp_lateral * lat
    hdg_term = 0.0 if hdg is None else config.kp_heading * np.deg2rad(hdg)
    steer = float(np.clip(lat_term + hdg_term, -config.max_steer, config.max_steer))
    lat_ok = lat is not None and abs(lat) <= config.deadband_m
    hdg_ok = hdg is not None and abs(hdg) <= config.deadband_deg
    if lat_ok and hdg_ok:
        hint, steer = "CENTER", 0.0
    elif steer > 0.04:
        hint = "LEFT"
    elif steer < -0.04:
        hint = "RIGHT"
    else:
        hint = "CENTER"
    half = 5
    bar = 0 if lat is None else int(np.clip(np.round((lat / max(config.lightbar_full_scale_m, 1e-3)) * half), -half, half))
    return SteerCommand(
        lat, hdg, steer, hint, bar, confidence=perception.confidence,
        reason="row_lock" if perception.left_line and perception.right_line else "partial",
    )
