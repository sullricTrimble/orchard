#include "orchard/visualize.hpp"
#include "orchard/geometry.hpp"
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <sstream>
namespace orchard {
static bool project_ground(float x, float z, float height, const Frame& frame, const Config& cfg, cv::Point& p) {
    float cx,cy,cz,u,v;
    ground_to_camera(x,height,z,cfg.pitch_rad(),cfg.camera_height_m,cx,cy,cz);
    if (cz<0.4f || !frame.K.cam_to_pixel(cx,cy,cz,u,v)) return false;
    p={ (int)std::lround(u), (int)std::lround(v) }; return true;
}
cv::Mat annotate(const Frame& frame, const RowPerception& perc, const SteerCommand& cmd,
                 const VelocityEstimate& vel, const Config& cfg) {
    cv::Mat rgb = frame.rgb.empty() ? cv::Mat(frame.K.height, frame.K.width, CV_8UC3, cv::Scalar(0,0,0)) : frame.rgb.clone();
    auto drawl=[&](const LineXZ& line, cv::Scalar color, float h=0.9f){
        cv::Point prev; bool have=false;
        for (int i=0;i<12;++i) {
            float z=1.6f+i*(16.4f/11.f); cv::Point p;
            if (!project_ground(line.x_at(z),z,h,frame,cfg,p)) continue;
            if (have) cv::line(rgb,prev,p,color,2,cv::LINE_AA); prev=p; have=true;
        }
    };
    if (perc.left_line) drawl(*perc.left_line,{80,180,255});
    if (perc.right_line) drawl(*perc.right_line,{80,180,255});
    if (perc.centerline) drawl(*perc.centerline,{60,220,90},0.2f);
    cv::Scalar color = cmd.hint=="CENTER" ? cv::Scalar(60,220,90) : cv::Scalar(80,180,255);
    cv::rectangle(rgb,{12,12},{420,118},{0,0,0},cv::FILLED);
    cv::putText(rgb,"STEER "+cmd.hint,{24,52},cv::FONT_HERSHEY_SIMPLEX,1.1,color,3);
    std::ostringstream ss; ss.setf(std::ios::fixed); ss.precision(2);
    ss<<"lat "<<(cmd.lateral_error_m?*cmd.lateral_error_m:0)<<"  trunks "<<perc.trunks.size();
    if (vel.speed_mps) ss<<"  "<<*vel.speed_mps<<" m/s";
    cv::putText(rgb,ss.str(),{24,90},cv::FONT_HERSHEY_SIMPLEX,0.5,{220,220,220},1);
    return rgb;
}
}
