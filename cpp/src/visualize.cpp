#include "orchard/visualize.hpp"
#include "orchard/geometry.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <opencv2/imgproc.hpp>
namespace orchard {
static cv::Scalar hint_color(const std::string& hint) {
    if (hint == "CENTER") return {60, 220, 90};
    if (hint == "HOLD") return {90, 90, 90};
    return {80, 180, 255};
}
static bool trunk_pixel(const Trunk& t, const Frame& frame, const Config& cfg, cv::Point& p) {
    float u, v;
    if (cfg.desk_mode) {
        if (!frame.K.cam_to_pixel(t.x_m, 0.f, t.z_m, u, v)) return false;
    } else {
        float cx, cy, cz;
        ground_to_camera(t.x_m, 0.8f, t.z_m, cfg.pitch_rad(), cfg.camera_height_m, cx, cy, cz);
        if (!frame.K.cam_to_pixel(cx, cy, cz, u, v)) return false;
    }
    p = {(int)std::lround(u), (int)std::lround(v)};
    return true;
}
static cv::Mat silhouette(const Frame& frame, const RowPerception& perc, const Config& cfg) {
    const int h = frame.depth_m.empty() ? std::max(1, frame.K.height) : frame.depth_m.rows;
    const int w = frame.depth_m.empty() ? std::max(1, frame.K.width) : frame.depth_m.cols;
    cv::Mat sil(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
    if (frame.depth_m.empty()) return sil;
    const bool left_ok = perc.left_line.has_value();
    const bool right_ok = perc.right_line.has_value();
    const float z0 = cfg.desk_mode ? cfg.desk_depth_min_m : cfg.depth_min_m;
    const float z1 = cfg.desk_mode ? cfg.desk_depth_max_m : cfg.depth_max_m;
    const int stride = std::max(1, cfg.pixel_stride);
    for (int v = 0; v < h; v += stride) {
        const float* row = frame.depth_m.ptr<float>(v);
        for (int u = 0; u < w; u += stride) {
            const float z = row[u];
            if (z < z0 || z > z1) continue;
            float x, y;
            frame.K.pixel_to_cam((float)u, (float)v, z, x, y);
            if (cfg.desk_mode) {
                if (std::abs(y) > cfg.desk_y_band_m) continue;
            } else {
                float xr, agl, zf;
                camera_point_to_ground(x, y, z, cfg.pitch_rad(), cfg.camera_height_m, xr, agl, zf);
                if (agl < cfg.trunk_height_min_m || agl > cfg.trunk_height_max_m) continue;
                x = xr;
            }
            cv::Vec3b c(70, 70, 70);
            if (x < -0.02f) c = left_ok ? cv::Vec3b(230, 220, 90) : cv::Vec3b(90, 90, 90);
            else if (x > 0.02f) c = right_ok ? cv::Vec3b(70, 165, 255) : cv::Vec3b(90, 90, 90);
            for (int dv = 0; dv < stride && v + dv < h; ++dv)
                for (int du = 0; du < stride && u + du < w; ++du)
                    sil.at<cv::Vec3b>(v + dv, u + du) = c;
        }
    }
    cv::dilate(sil, sil, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
    const int cx = (int)std::lround(frame.K.cx > 0 ? frame.K.cx : w * 0.5f);
    cv::line(sil, {cx, 0}, {cx, h - 1}, {50, 50, 50}, 1, cv::LINE_AA);
    for (const auto& t : perc.trunks) {
        cv::Point p;
        if (!trunk_pixel(t, frame, cfg, p)) continue;
        cv::Scalar ring = t.side == Trunk::Left ? cv::Scalar(230, 220, 90) : cv::Scalar(70, 165, 255);
        cv::circle(sil, p, 16, ring, 2, cv::LINE_AA);
    }
    return sil;
}
cv::Mat annotate(const Frame& frame, const RowPerception& perc, const SteerCommand& cmd,
                 const VelocityEstimate& vel, const Config& cfg) {
    cv::Mat sil = silhouette(frame, perc, cfg);
    const int w = sil.cols, banner_h = 150;
    cv::Mat out(sil.rows + banner_h, w, CV_8UC3, cv::Scalar(8, 8, 8));
    sil.copyTo(out(cv::Rect(0, banner_h, w, sil.rows)));
    const cv::Scalar hc = hint_color(cmd.hint);
    cv::rectangle(out, {0, 0}, {w, banner_h}, {12, 12, 12}, cv::FILLED);
    cv::rectangle(out, {0, banner_h - 4}, {w, banner_h}, hc, cv::FILLED);
    std::string title = cmd.hint == "LEFT" ? "<  LEFT" : (cmd.hint == "RIGHT" ? "RIGHT  >" : cmd.hint);
    int baseline = 0;
    cv::Size ts = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 2.2, 5, &baseline);
    cv::putText(out, title, {(w - ts.width) / 2, 78}, cv::FONT_HERSHEY_SIMPLEX, 2.2, hc, 5, cv::LINE_AA);
    const int half = 5, cell_w = 28, cell_h = 22, gap = 6;
    const int bar_w = 11 * cell_w + 10 * gap;
    int x0 = (w - bar_w) / 2, y0 = 104;
    for (int i = -half; i <= half; ++i) {
        cv::Rect r(x0 + (i + half) * (cell_w + gap), y0, cell_w, cell_h);
        bool on = (i == 0 && cmd.hint == "CENTER");
        if (cmd.lightbar > 0 && i < 0 && i >= -cmd.lightbar) on = true;
        if (cmd.lightbar < 0 && i > 0 && i <= -cmd.lightbar) on = true;
        cv::Scalar fill = on ? (i == 0 ? cv::Scalar(60, 220, 90) : hc) : cv::Scalar(28, 28, 28);
        cv::rectangle(out, r, fill, cv::FILLED);
        cv::rectangle(out, r, {50, 50, 50}, 1);
    }
    char line[160];
    if (cmd.lateral_error_m)
        std::snprintf(line, sizeof(line), "lat %+0.2f m   trunks %d   conf %0.2f",
                      *cmd.lateral_error_m, (int)perc.trunks.size(), perc.confidence);
    else
        std::snprintf(line, sizeof(line), "lat --   trunks %d   conf %0.2f",
                      (int)perc.trunks.size(), perc.confidence);
    if (vel.speed_mps) {
        char extra[40];
        std::snprintf(extra, sizeof(extra), "   %0.2f m/s", *vel.speed_mps);
        std::strncat(line, extra, sizeof(line) - std::strlen(line) - 1);
    }
    cv::putText(out, line, {16, banner_h + sil.rows - 16}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                {180, 180, 180}, 1, cv::LINE_AA);
    return out;
}
}
