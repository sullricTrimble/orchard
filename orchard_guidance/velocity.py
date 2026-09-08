from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import GuidanceConfig
from .types import Trunk


@dataclass
class TrackedTrunk:
    tid: int
    side: str
    x_m: float
    z_m: float
    t_s: float
    missed: int = 0


@dataclass
class VelocityEstimate:
    speed_mps: Optional[float] = None
    speed_from_tracks_mps: Optional[float] = None
    speed_from_spacing_mps: Optional[float] = None
    measured_spacing_m: Optional[float] = None
    trees_passed: int = 0
    tracks: int = 0
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


class VelocityEstimator:
    def __init__(self, config: GuidanceConfig):
        self.config = config
        self.tracks: list[TrackedTrunk] = []
        self.next_id = 1
        self.smoothed_speed: Optional[float] = None
        self.gate_last_t: dict[str, Optional[float]] = {"left": None, "right": None}
        self.trees_passed = 0
        self.pass_intervals: list[float] = []

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1
        self.smoothed_speed = None
        self.gate_last_t = {"left": None, "right": None}
        self.trees_passed = 0
        self.pass_intervals = []

    def update(self, trunks: list[Trunk], timestamp_s: float) -> VelocityEstimate:
        cfg = self.config
        inst: list[float] = []
        unmatched = list(trunks)
        for track in self.tracks:
            track.missed += 1
        for trunk in list(unmatched):
            best = None
            best_dz = 1e9
            for track in self.tracks:
                if track.side != trunk.side:
                    continue
                dz = track.z_m - trunk.z_m
                if dz < -0.25 or dz > cfg.max_match_dz_m:
                    continue
                dx = abs(track.x_m - trunk.x_m)
                if dx > 1.2:
                    continue
                score = abs(dz) + 0.3 * dx
                if score < best_dz:
                    best_dz = score
                    best = track
            if best is None:
                continue
            dt = timestamp_s - best.t_s
            if dt >= cfg.min_track_dt_s:
                v = (best.z_m - trunk.z_m) / dt
                if 0.05 < v < 8.0:
                    inst.append(v)
            gate = cfg.speed_gate_m
            if best.z_m >= gate > trunk.z_m:
                last = self.gate_last_t[trunk.side]
                if last is not None:
                    interval = timestamp_s - last
                    if 0.25 < interval < 12.0:
                        self.pass_intervals.append(interval)
                self.gate_last_t[trunk.side] = timestamp_s
                self.trees_passed += 1
            best.x_m, best.z_m, best.t_s, best.missed = trunk.x_m, trunk.z_m, timestamp_s, 0
            unmatched.remove(trunk)

        self.tracks = [t for t in self.tracks if t.missed < 6]
        for trunk in unmatched:
            self.tracks.append(
                TrackedTrunk(tid=self.next_id, side=trunk.side, x_m=trunk.x_m, z_m=trunk.z_m, t_s=timestamp_s)
            )
            self.next_id += 1

        out = VelocityEstimate(tracks=len(self.tracks), trees_passed=self.trees_passed)
        if inst:
            out.speed_from_tracks_mps = float(np.median(inst))
        spacings = []
        for side in ("left", "right"):
            zs = sorted(t.z_m for t in trunks if t.side == side)
            spacings.extend(float(b - a) for a, b in zip(zs, zs[1:]) if 1.2 < (b - a) < 8.0)
        if spacings:
            out.measured_spacing_m = float(np.median(spacings))
        spacing = out.measured_spacing_m or cfg.tree_spacing_m
        if self.pass_intervals:
            out.speed_from_spacing_mps = float(spacing / np.median(self.pass_intervals[-6:]))
        fused = None
        if out.speed_from_tracks_mps is not None and out.speed_from_spacing_mps is not None:
            fused = 0.7 * out.speed_from_tracks_mps + 0.3 * out.speed_from_spacing_mps
        elif out.speed_from_tracks_mps is not None:
            fused = out.speed_from_tracks_mps
        elif out.speed_from_spacing_mps is not None:
            fused = out.speed_from_spacing_mps
        if fused is not None:
            a = cfg.speed_smooth
            self.smoothed_speed = fused if self.smoothed_speed is None else (1.0 - a) * self.smoothed_speed + a * fused
            out.speed_mps = float(self.smoothed_speed)
        conf = 0.0
        if out.speed_from_tracks_mps is not None:
            conf += 0.55
        if out.speed_from_spacing_mps is not None:
            conf += 0.30
        if out.tracks >= 4:
            conf += 0.15
        out.confidence = float(min(1.0, conf))
        return out
