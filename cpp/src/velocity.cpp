#include "orchard/velocity.hpp"
#include <algorithm>
#include <cmath>
namespace orchard {
VelocityEstimator::VelocityEstimator(Config cfg) : cfg_(std::move(cfg)) {}
void VelocityEstimator::reset() {
    tracks_.clear(); next_id_ = 1; smoothed_.reset(); gate_left_.reset(); gate_right_.reset();
    trees_passed_ = 0; pass_intervals_.clear();
}
static float median(std::vector<float> v) {
    if (v.empty()) return 0.f;
    std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
    return v[v.size()/2];
}
VelocityEstimate VelocityEstimator::update(const std::vector<Trunk>& trunks, double timestamp_s) {
    std::vector<float> inst;
    for (auto& t : tracks_) t.missed++;
    std::vector<char> matched(trunks.size(), 0);
    for (size_t i = 0; i < trunks.size(); ++i) {
        const Trunk& trunk = trunks[i];
        Tracked* best = nullptr; float best_score = 1e9f;
        for (auto& track : tracks_) {
            if (track.side != trunk.side) continue;
            const float dz = track.z_m - trunk.z_m;
            if (dz < -0.25f || dz > cfg_.max_match_dz_m) continue;
            const float dx = std::abs(track.x_m - trunk.x_m);
            if (dx > 1.2f) continue;
            const float score = std::abs(dz) + 0.3f * dx;
            if (score < best_score) { best_score = score; best = &track; }
        }
        if (!best) continue;
        const double dt = timestamp_s - best->t_s;
        if (dt >= cfg_.min_track_dt_s) {
            const float v = static_cast<float>((best->z_m - trunk.z_m) / dt);
            if (v > 0.05f && v < 8.f) inst.push_back(v);
        }
        if (best->z_m >= cfg_.speed_gate_m && trunk.z_m < cfg_.speed_gate_m) {
            auto& gate = trunk.side == Trunk::Left ? gate_left_ : gate_right_;
            if (gate) {
                const float interval = static_cast<float>(timestamp_s - *gate);
                if (interval > 0.25f && interval < 12.f) pass_intervals_.push_back(interval);
            }
            gate = timestamp_s; trees_passed_++;
        }
        best->x_m = trunk.x_m; best->z_m = trunk.z_m; best->t_s = timestamp_s; best->missed = 0;
        matched[i] = 1;
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [](const Tracked& t){ return t.missed >= 6; }), tracks_.end());
    for (size_t i = 0; i < trunks.size(); ++i)
        if (!matched[i]) tracks_.push_back({next_id_++, trunks[i].side, trunks[i].x_m, trunks[i].z_m, timestamp_s, 0});
    VelocityEstimate out; out.tracks = (int)tracks_.size(); out.trees_passed = trees_passed_;
    if (!inst.empty()) out.speed_from_tracks_mps = median(inst);
    std::vector<float> spacings;
    for (auto side : {Trunk::Left, Trunk::Right}) {
        std::vector<float> zs;
        for (const auto& t : trunks) if (t.side == side) zs.push_back(t.z_m);
        std::sort(zs.begin(), zs.end());
        for (size_t i = 1; i < zs.size(); ++i) { float d = zs[i]-zs[i-1]; if (d>1.2f && d<8.f) spacings.push_back(d); }
    }
    if (!spacings.empty()) out.measured_spacing_m = median(spacings);
    const float spacing = out.measured_spacing_m.value_or(cfg_.tree_spacing_m);
    if (!pass_intervals_.empty()) {
        int n = std::min(6, (int)pass_intervals_.size());
        std::vector<float> recent(pass_intervals_.end()-n, pass_intervals_.end());
        out.speed_from_spacing_mps = spacing / median(recent);
    }
    std::optional<float> fused;
    if (out.speed_from_tracks_mps && out.speed_from_spacing_mps)
        fused = 0.7f * *out.speed_from_tracks_mps + 0.3f * *out.speed_from_spacing_mps;
    else if (out.speed_from_tracks_mps) fused = out.speed_from_tracks_mps;
    else if (out.speed_from_spacing_mps) fused = out.speed_from_spacing_mps;
    if (fused) {
        float a = cfg_.speed_smooth;
        smoothed_ = smoothed_ ? (1.f-a)* *smoothed_ + a* *fused : fused;
        out.speed_mps = smoothed_;
    }
    if (out.speed_from_tracks_mps) out.confidence += 0.55f;
    if (out.speed_from_spacing_mps) out.confidence += 0.30f;
    if (out.tracks >= 4) out.confidence += 0.15f;
    out.confidence = std::min(1.f, out.confidence);
    return out;
}
}
