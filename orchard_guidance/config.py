from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GuidanceConfig:
    """Physical and algorithm knobs for an orchard-row prototype."""

    row_width_m: float = 4.5
    tree_spacing_m: float = 3.5
    trunk_radius_m: float = 0.12

    camera_height_m: float = 1.55
    camera_pitch_deg: float = -6.0  # negative = look slightly down
    image_width: int = 848
    image_height: int = 480
    depth_hfov_deg: float = 87.0
    depth_vfov_deg: float = 58.0

    trunk_height_min_m: float = 0.40
    trunk_height_max_m: float = 1.20
    depth_min_m: float = 1.2
    depth_max_m: float = 12.0
    center_keepout_m: float = 0.45
    cluster_eps_m: float = 0.50
    cluster_min_points: int = 10
    ransac_thresh_m: float = 0.40
    min_trunks_for_line: int = 2

    kp_lateral: float = 0.40
    kp_heading: float = 1.00
    deadband_m: float = 0.10
    deadband_deg: float = 2.5
    lightbar_full_scale_m: float = 0.80
    max_steer: float = 1.0

    speed_gate_m: float = 3.2
    speed_smooth: float = 0.25
    min_track_dt_s: float = 0.04
    max_match_dz_m: float = 1.6

    sim_row_length_m: float = 70.0
    sim_spacing_jitter_m: float = 0.12
    sim_lateral_jitter_m: float = 0.08
    sim_fps: float = 15.0
    sim_speed_mps: float = 1.8
    sim_start_lateral_m: float = 0.65
    sim_start_yaw_deg: float = 7.0

    extra: dict = field(default_factory=dict)

    # Desk bring-up: two pens held in front of the D455 instead of orchard trunks.
    desk_mode: bool = False
    desk_depth_min_m: float = 0.30
    desk_depth_max_m: float = 2.20
    desk_y_band_m: float = 0.28
    desk_cluster_eps_m: float = 0.09
    desk_cluster_min_points: int = 12
    desk_max_span_m: float = 0.22
    desk_min_vertical_m: float = 0.05
    desk_min_gap_m: float = 0.08
    desk_max_gap_m: float = 1.20
    desk_max_lat_m: float = 0.45
    print_period_s: float = 0.75

    @property
    def camera_pitch_rad(self) -> float:
        return float(self.camera_pitch_deg) * 3.141592653589793 / 180.0

    def apply_desk_pens(self) -> "GuidanceConfig":
        """Tune knobs so two handheld pens read as the left/right alley edges."""
        self.desk_mode = True
        self.camera_pitch_deg = 0.0
        self.lightbar_full_scale_m = 0.20
        self.deadband_m = 0.03
        self.kp_lateral = 1.20
        self.kp_heading = 0.0
        self.depth_min_m = self.desk_depth_min_m
        self.depth_max_m = self.desk_depth_max_m
        return self
