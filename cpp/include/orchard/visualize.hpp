#pragma once
#include "orchard/config.hpp"
#include "orchard/types.hpp"
namespace orchard {
cv::Mat annotate(const Frame& frame, const RowPerception& perc, const SteerCommand& cmd,
                 const VelocityEstimate& vel, const Config& cfg);
}
