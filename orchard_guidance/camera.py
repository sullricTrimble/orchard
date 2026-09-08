from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from .config import GuidanceConfig
from .types import Frame, Intrinsics


class FrameSource(Protocol):
    def read(self) -> Optional[Frame]:
        ...

    def close(self) -> None:
        ...


class RealSenseSource:
    """Live Intel RealSense D455 (or playback of a .bag recording)."""

    def __init__(self, bag_path: Optional[str] = None, config: Optional[GuidanceConfig] = None):
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. On the machine with the D455 run:\n"
                "  pip install pyrealsense2\n"
                "Alternatively use --source sim."
            ) from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if bag_path:
            cfg.enable_device_from_file(bag_path, repeat_playback=True)
        else:
            cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        self._align = rs.align(rs.stream.color)
        profile = self._pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._scale = float(depth_sensor.get_depth_scale())
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.intrinsics = Intrinsics(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
            width=int(intr.width),
            height=int(intr.height),
        )
        self._t0: Optional[float] = None

    def read(self) -> Optional[Frame]:
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()
        if not depth or not color:
            return None
        rgb = np.asanyarray(color.get_data())
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self._scale
        ts = float(color.get_timestamp()) / 1000.0
        if self._t0 is None:
            self._t0 = ts
        return Frame(rgb=rgb, depth_m=depth_m, intrinsics=self.intrinsics, timestamp_s=ts - self._t0)

    def close(self) -> None:
        self._pipeline.stop()
