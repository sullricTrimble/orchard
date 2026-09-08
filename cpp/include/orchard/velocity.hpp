#pragma once
#include "orchard/config.hpp"
#include "orchard/types.hpp"
namespace orchard {
class VelocityEstimator {
public:
    explicit VelocityEstimator(Config cfg);
    void reset();
    VelocityEstimate update(const std::vector<Trunk>& trunks, double timestamp_s);
private:
    struct Tracked { int id=0; Trunk::Side side=Trunk::Left; float x_m=0, z_m=0; double t_s=0; int missed=0; };
    Config cfg_; std::vector<Tracked> tracks_; int next_id_=1;
    std::optional<float> smoothed_; std::optional<double> gate_left_, gate_right_;
    int trees_passed_=0; std::vector<float> pass_intervals_;
};
}
