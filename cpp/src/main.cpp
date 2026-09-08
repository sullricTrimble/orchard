#include "orchard/config.hpp"
#include "orchard/guidance.hpp"
#include "orchard/perception.hpp"
#include "orchard/synthetic.hpp"
#include "orchard/velocity.hpp"
#include "orchard/visualize.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#ifdef HAVE_REALSENSE
#include <librealsense2/rs.hpp>
#endif
using namespace orchard;
int main(int argc, char** argv) {
    std::string source="sim"; bool gui=false; int max_frames=90;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"--source") && i+1<argc) source=argv[++i];
        else if (!strcmp(argv[i],"--gui")) gui=true;
        else if (!strcmp(argv[i],"--no-gui")) gui=false;
        else if (!strcmp(argv[i],"--frames") && i+1<argc) max_frames=atoi(argv[++i]);
    }
    Config cfg;
    if (source=="sim") {
        auto world=build_world(cfg); TractorPose pose; pose.x_m=0.65f; pose.yaw_rad=7.f*0.0174533f; pose.s_m=6.f;
        VelocityEstimator vel(cfg); float dt=1.f/15.f; double t=0;
        for (int i=0;i<max_frames;++i) {
            Frame frame=render_frame(world,pose,cfg,t);
            auto perc=perceive(frame,cfg); auto cmd=compute_steer(perc,cfg); auto speed=vel.update(perc.trunks,t);
            auto vis=annotate(frame,perc,cmd,speed,cfg);
            if (gui) { cv::imshow("Orchard", vis); if (cv::waitKey(1)==27) break; }
            else if (i%15==0) std::cout<<"t="<<t<<" "<<cmd.hint<<" lat="<<(cmd.lateral_error_m?*cmd.lateral_error_m:0)<<"\n";
            pose.yaw_rad += (-cmd.steer*0.42f)*dt;
            pose.x_m += pose.speed_mps*std::sin(pose.yaw_rad)*dt;
            pose.s_m += pose.speed_mps*std::cos(pose.yaw_rad)*dt;
            t+=dt;
        }
        return 0;
    }
#ifdef HAVE_REALSENSE
    if (source=="realsense") {
        rs2::config rs_cfg;
        rs_cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 15);
        rs_cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 15);
        rs2::pipeline pipe; auto profile=pipe.start(rs_cfg);
        rs2::decimation_filter dec; dec.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
        float scale=profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
        auto intr=profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
        VelocityEstimator vel(cfg);
        while (true) {
            auto frames=pipe.wait_for_frames();
            rs2::depth_frame depth=dec.process(frames.get_depth_frame());
            auto color=frames.get_color_frame();
            Frame frame;
            frame.K.fx=intr.fx*depth.get_width()/(float)intr.width;
            frame.K.fy=intr.fy*depth.get_height()/(float)intr.height;
            frame.K.cx=intr.ppx*depth.get_width()/(float)intr.width;
            frame.K.cy=intr.ppy*depth.get_height()/(float)intr.height;
            frame.K.width=depth.get_width(); frame.K.height=depth.get_height();
            cv::Mat depth16(depth.get_height(), depth.get_width(), CV_16UC1, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
            depth16.convertTo(frame.depth_m, CV_32F, scale);
            cv::Mat color_mat(color.get_height(), color.get_width(), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
            cv::resize(color_mat, frame.rgb, frame.depth_m.size());
            auto perc=perceive(frame,cfg); auto cmd=compute_steer(perc,cfg);
            auto vis=annotate(frame,perc,cmd,vel.update(perc.trunks,0),cfg);
            if (gui) { cv::imshow("Orchard", vis); if (cv::waitKey(1)==27) break; }
            std::cout<<"\r"<<cmd.hint<<" lat="<<(cmd.lateral_error_m?*cmd.lateral_error_m:0)<<"   "<<std::flush;
        }
        return 0;
    }
#else
    if (source=="realsense") { std::cerr<<"Built without librealsense2\n"; return 1; }
#endif
    std::cerr<<"Unknown source\n"; return 1;
}
