#include "orchard/config.hpp"
#include "orchard/geometry.hpp"
#include "orchard/guidance.hpp"
#include "orchard/perception.hpp"
#include "orchard/synthetic.hpp"
#include <cmath>
#include <iostream>
using namespace orchard;
static int fail(const char* m){ std::cerr<<"FAIL: "<<m<<"\n"; return 1; }
int main() {
    Config cfg; cfg.image_width=640; cfg.image_height=360; cfg.pixel_stride=1;
    float pitch=-6.f*0.0174533f, x,y,z,xr,agl,zf;
    ground_to_camera(-2.25f,0.8f,6.f,pitch,1.55f,x,y,z);
    camera_point_to_ground(x,y,z,pitch,1.55f,xr,agl,zf);
    if (std::abs(xr+2.25f)>0.01f) return fail("geometry");
    auto world=build_world(cfg,1); TractorPose pose; pose.s_m=8.f; pose.speed_mps=0;
    pose.x_m=0.70f;
    auto perc=perceive(render_frame(world,pose,cfg,0),cfg);
    auto cmd=compute_steer(perc,cfg);
    if (!cmd.lateral_error_m || *cmd.lateral_error_m<0.25f) return fail("right offset");
    if (cmd.hint!="LEFT") return fail("steer left");
    pose.x_m=-0.70f;
    perc=perceive(render_frame(world,pose,cfg,0),cfg); cmd=compute_steer(perc,cfg);
    if (!cmd.lateral_error_m || *cmd.lateral_error_m>-0.25f) return fail("left offset");
    std::cout<<"C++ guidance smoke tests passed.\n";
    return 0;
}
