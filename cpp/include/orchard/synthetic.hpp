#pragma once
#include <cstdint>
#include <vector>
#include <opencv2/core.hpp>
#include "orchard/config.hpp"
#include "orchard/types.hpp"
namespace orchard {
struct TractorPose { float x_m=0, s_m=6, yaw_rad=0, speed_mps=1.8f; };
struct OrchardWorld {
    std::vector<cv::Point2f> left_trees, right_trees;
    float row_width_m=4.5f, end_z_m=70.f;
};
OrchardWorld build_world(const Config& cfg, uint32_t seed=3);
Frame render_frame(const OrchardWorld& world, const TractorPose& pose, const Config& cfg, double timestamp_s);
}
