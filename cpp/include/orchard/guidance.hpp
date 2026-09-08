#pragma once
#include "orchard/config.hpp"
#include "orchard/types.hpp"
namespace orchard { SteerCommand compute_steer(const RowPerception& perc, const Config& cfg); }
