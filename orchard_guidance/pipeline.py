from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from .config import GuidanceConfig
from .guidance import SteerCommand, compute_steer
from .perception import perceive
from .types import Frame, RowPerception
from .velocity import VelocityEstimate, VelocityEstimator


def _smooth(prev, new, alpha):
    if new is None:
        return prev
    if prev is None:
        return new
    return (1.0 - alpha) * prev + alpha * new


@dataclass
class GuidanceOutput:
    timestamp_s: float
    lateral_error_m: Optional[float]
    heading_error_deg: Optional[float]
    steer: float
    hint: str
    lightbar: int
    confidence: float
    speed_mps: Optional[float]
    speed_confidence: float
    measured_spacing_m: Optional[float]
    measured_row_width_m: Optional[float]
    trunks: int
    trees_passed: int
    vanishing_point: Optional[tuple[float, float]]
    notes: list[str]
    source: str
    desk_mode: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class GuidancePipeline:
    def __init__(self, config: Optional[GuidanceConfig] = None, source_name: str = "sim"):
        self.config = config or GuidanceConfig()
        self.source_name = source_name
        self.velocity = VelocityEstimator(self.config)
        self._lat = None
        self._hdg = None
        self.last_perception: Optional[RowPerception] = None
        self.last_command: Optional[SteerCommand] = None
        self.last_speed: Optional[VelocityEstimate] = None
        self.last_output: Optional[GuidanceOutput] = None

    def reset(self) -> None:
        self.velocity.reset()
        self._lat = self._hdg = None
        self.last_perception = self.last_command = self.last_speed = self.last_output = None

    def process(self, frame: Frame) -> GuidanceOutput:
        perc = perceive(frame, self.config)
        if perc.lateral_error_m is None and self.config.desk_mode:
            self._lat = self._hdg = None
        else:
            self._lat = _smooth(self._lat, perc.lateral_error_m, 0.35)
            hdg = None if perc.heading_error_rad is None else float(perc.heading_error_rad)
            self._hdg = _smooth(self._hdg, hdg, 0.35)
        perc.lateral_error_m = self._lat
        perc.heading_error_rad = self._hdg
        cmd = compute_steer(perc, self.config)
        speed = self.velocity.update(perc.trunks, frame.timestamp_s)
        self.last_perception, self.last_command, self.last_speed = perc, cmd, speed
        out = GuidanceOutput(
            timestamp_s=frame.timestamp_s,
            lateral_error_m=cmd.lateral_error_m,
            heading_error_deg=cmd.heading_error_deg,
            steer=cmd.steer,
            hint=cmd.hint,
            lightbar=cmd.lightbar,
            confidence=cmd.confidence,
            speed_mps=speed.speed_mps,
            speed_confidence=speed.confidence,
            measured_spacing_m=speed.measured_spacing_m,
            measured_row_width_m=perc.row_width_m,
            trunks=len(perc.trunks),
            trees_passed=speed.trees_passed,
            vanishing_point=perc.vanishing_point_uv,
            notes=list(perc.notes),
            source=self.source_name,
            desk_mode=self.config.desk_mode,
        )
        self.last_output = out
        return out
