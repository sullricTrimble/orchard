#include "orchard/config.hpp"
#include "orchard/guidance.hpp"
#include "orchard/perception.hpp"
#include "orchard/synthetic.hpp"
#include "orchard/velocity.hpp"
#include "orchard/visualize.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef HAVE_REALSENSE
#include <librealsense2/rs.hpp>
#endif
using namespace orchard;

static bool arg_eq(const char* a, const char* b) { return a && b && !std::strcmp(a, b); }

static void print_live(const SteerCommand& cmd, const RowPerception& perc) {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(2);
    std::cout << "LIVE " << cmd.hint;
    std::cout << "  lat=";
    if (cmd.lateral_error_m) std::cout << (*cmd.lateral_error_m >= 0 ? "+" : "") << *cmd.lateral_error_m;
    else std::cout << "--";
    std::cout << "  pens=" << perc.trunks.size();
    std::cout << "  conf=" << perc.confidence;
    for (const auto& n : perc.notes) std::cout << "  " << n;
    std::cout << "\n";
}

int main(int argc, char** argv) {
    std::string source="sim"; bool gui=false; int max_frames=90;
    Config cfg;
    for (int i=1;i<argc;++i) {
        if (arg_eq(argv[i],"--source") && i+1<argc) source=argv[++i];
        else if (arg_eq(argv[i],"--gui")) gui=true;
        else if (arg_eq(argv[i],"--no-gui")) gui=false;
        else if (arg_eq(argv[i],"--frames") && i+1<argc) max_frames=atoi(argv[++i]);
        else if (arg_eq(argv[i],"--pens")) cfg.apply_desk_pens();
        else if (arg_eq(argv[i],"--row-width") && i+1<argc) cfg.row_width_m=(float)atof(argv[++i]);
        else if (arg_eq(argv[i],"--tree-spacing") && i+1<argc) cfg.tree_spacing_m=(float)atof(argv[++i]);
        else if (arg_eq(argv[i],"--trunk-min") && i+1<argc) cfg.trunk_height_min_m=(float)atof(argv[++i]);
        else if (arg_eq(argv[i],"--trunk-max") && i+1<argc) cfg.trunk_height_max_m=(float)atof(argv[++i]);
        else if (arg_eq(argv[i],"--camera-height") && i+1<argc) cfg.camera_height_m=(float)atof(argv[++i]);
        else if (arg_eq(argv[i],"--print-period") && i+1<argc) cfg.print_period_s=(float)atof(argv[++i]);
    }
    if (source=="sim") {
        auto world=build_world(cfg); TractorPose pose; pose.x_m=0.65f; pose.yaw_rad=7.f*0.0174533f; pose.s_m=6.f;
        VelocityEstimator vel(cfg); float dt=1.f/15.f; double t=0;
        auto last_print=std::chrono::steady_clock::now();
        for (int i=0;i<max_frames;++i) {
            Frame frame=render_frame(world,pose,cfg,t);
            auto perc=perceive(frame,cfg); auto cmd=compute_steer(perc,cfg); auto speed=vel.update(perc.trunks,t);
            auto vis=annotate(frame,perc,cmd,speed,cfg);
            if (gui) { cv::imshow("Orchard", vis); if (cv::waitKey(1)==27) break; }
            auto now=std::chrono::steady_clock::now();
            if (std::chrono::duration<float>(now-last_print).count()>=cfg.print_period_s) {
                print_live(cmd, perc);
                last_print=now;
            }
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
        float scale=profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
        auto intr=profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
        VelocityEstimator vel(cfg);
        auto last_print=std::chrono::steady_clock::now();
        while (true) {
            rs2::frameset frames;
            if (!pipe.poll_for_frames(&frames)) {
                if (gui && cv::waitKey(5)==27) break;
                continue;
            }
            rs2::depth_frame depth=frames.get_depth_frame();
            auto color=frames.get_color_frame();
            if (!depth) continue;
            Frame frame;
            frame.K.fx=intr.fx; frame.K.fy=intr.fy; frame.K.cx=intr.ppx; frame.K.cy=intr.ppy;
            frame.K.width=depth.get_width(); frame.K.height=depth.get_height();
            cv::Mat depth16(depth.get_height(), depth.get_width(), CV_16UC1, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
            depth16.convertTo(frame.depth_m, CV_32F, scale);
            if (cv::mean(frame.depth_m)[0] <= 0) continue;
            if (color) {
                cv::Mat color_mat(color.get_height(), color.get_width(), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
                cv::resize(color_mat, frame.rgb, frame.depth_m.size());
            }
            auto perc=perceive(frame,cfg); auto cmd=compute_steer(perc,cfg);
            auto vis=annotate(frame,perc,cmd,vel.update(perc.trunks,0),cfg);
            if (gui) { cv::imshow("Orchard", vis); if (cv::waitKey(1)==27) break; }
            auto now=std::chrono::steady_clock::now();
            if (std::chrono::duration<float>(now-last_print).count()>=cfg.print_period_s) {
                print_live(cmd, perc);
                last_print=now;
            }
        }
        return 0;
    }
#else
    if (source=="realsense") { std::cerr<<"Built without librealsense2\n"; return 1; }
#endif
    std::cerr<<"Unknown source\n"; return 1;
}
