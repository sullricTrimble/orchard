#pragma once
#include <cmath>
#include <optional>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
namespace orchard {
struct Intrinsics {
    float fx = 1.f, fy = 1.f, cx = 0.f, cy = 0.f;
    int width = 0, height = 0;
    static Intrinsics from_fov(int w, int h, float hfov_deg, float vfov_deg) {
        Intrinsics k;
        k.width = w; k.height = h;
        k.fx = w / (2.f * std::tan(hfov_deg * 0.01745329252f / 2.f));
        k.fy = h / (2.f * std::tan(vfov_deg * 0.01745329252f / 2.f));
        k.cx = w * 0.5f; k.cy = h * 0.5f;
        return k;
    }
    void pixel_to_cam(float u, float v, float z, float& x, float& y) const {
        x = (u - cx) * z / fx; y = (v - cy) * z / fy;
    }
    bool cam_to_pixel(float x, float y, float z, float& u, float& v) const {
        if (z < 1e-4f) return false;
        u = fx * x / z + cx; v = fy * y / z + cy;
        return std::isfinite(u) && std::isfinite(v);
    }
};
struct Frame { cv::Mat rgb, depth_m; Intrinsics K; double timestamp_s = 0.0; };
struct LineXZ {
    float intercept = 0.f, slope = 0.f;
    float x_at(float z) const { return intercept + slope * z; }
    float heading_rad() const { return std::atan(slope); }
};
struct Trunk { enum Side { Left, Right } side = Left; float x_m = 0.f, z_m = 0.f; int n_points = 0; };
struct RowPerception {
    std::optional<LineXZ> left_line, right_line, centerline;
    std::optional<float> lateral_error_m, heading_error_rad, heading_from_rgb_rad, row_width_m;
    std::optional<cv::Point2f> vanishing_point_uv;
    std::vector<Trunk> trunks;
    float confidence = 0.f;
    std::vector<std::string> notes;
};
struct SteerCommand {
    std::optional<float> lateral_error_m, heading_error_deg;
    float steer = 0.f; std::string hint = "HOLD"; int lightbar = 0; float confidence = 0.f;
};
struct VelocityEstimate {
    std::optional<float> speed_mps, speed_from_tracks_mps, speed_from_spacing_mps, measured_spacing_m;
    int trees_passed = 0, tracks = 0; float confidence = 0.f;
};
}
