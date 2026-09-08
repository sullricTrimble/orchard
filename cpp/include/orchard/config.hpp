#pragma once
namespace orchard {
struct Config {
    float row_width_m = 4.5f, tree_spacing_m = 3.5f;
    float camera_height_m = 1.55f, camera_pitch_deg = -6.0f;
    int image_width = 848, image_height = 480, fps = 15;
    float depth_hfov_deg = 87.0f, depth_vfov_deg = 58.0f;
    float trunk_height_min_m = 0.40f, trunk_height_max_m = 1.20f;
    float depth_min_m = 1.2f, depth_max_m = 12.0f, center_keepout_m = 0.45f;
    float cluster_eps_m = 0.50f, ransac_thresh_m = 0.40f;
    int cluster_min_points = 10, min_trunks_for_line = 2, pixel_stride = 2;
    float kp_lateral = 0.40f, kp_heading = 1.00f, deadband_m = 0.10f, deadband_deg = 2.5f;
    float lightbar_full_scale_m = 0.80f, max_steer = 1.0f;
    float speed_gate_m = 3.2f, speed_smooth = 0.25f, min_track_dt_s = 0.04f, max_match_dz_m = 1.6f;
    float pitch_rad() const { return camera_pitch_deg * 0.01745329252f; }
};
}
