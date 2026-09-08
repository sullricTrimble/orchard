#include "orchard/guidance.hpp"
#include <algorithm>
#include <cmath>
namespace orchard {
SteerCommand compute_steer(const RowPerception& perc, const Config& cfg) {
    SteerCommand cmd;
    cmd.lateral_error_m = perc.lateral_error_m;
    if (perc.heading_error_rad) cmd.heading_error_deg = *perc.heading_error_rad * 57.2957795f;
    cmd.confidence = perc.confidence;
    if (!perc.lateral_error_m && !perc.heading_error_rad) { cmd.hint = "HOLD"; return cmd; }
    const float lat_term = perc.lateral_error_m ? cfg.kp_lateral * *perc.lateral_error_m : 0.f;
    const float hdg_term = perc.heading_error_rad ? cfg.kp_heading * *perc.heading_error_rad : 0.f;
    cmd.steer = std::clamp(lat_term + hdg_term, -cfg.max_steer, cfg.max_steer);
    const bool lat_ok = perc.lateral_error_m && std::abs(*perc.lateral_error_m) <= cfg.deadband_m;
    const bool hdg_ok = cmd.heading_error_deg && std::abs(*cmd.heading_error_deg) <= cfg.deadband_deg;
    if (lat_ok && hdg_ok) { cmd.hint = "CENTER"; cmd.steer = 0.f; }
    else if (cmd.steer > 0.04f) cmd.hint = "LEFT";
    else if (cmd.steer < -0.04f) cmd.hint = "RIGHT";
    else cmd.hint = "CENTER";
    if (perc.lateral_error_m) {
        const float n = *perc.lateral_error_m / std::max(cfg.lightbar_full_scale_m, 1e-3f);
        cmd.lightbar = static_cast<int>(std::clamp(std::round(n * 5.f), -5.f, 5.f));
    }
    return cmd;
}
}
