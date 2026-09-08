#include "orchard/synthetic.hpp"
#include "orchard/geometry.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <random>
#include <utility>
namespace orchard {
OrchardWorld build_world(const Config& cfg, uint32_t seed) {
    OrchardWorld w; w.row_width_m=cfg.row_width_m;
    std::mt19937 rng(seed); std::uniform_real_distribution<float> jz(-0.12f,0.12f), jx(-0.08f,0.08f);
    float half=cfg.row_width_m*0.5f;
    for (float z=2.f; z<70.f; z+=cfg.tree_spacing_m) {
        float zz=z+jz(rng);
        w.left_trees.emplace_back(-half+jx(rng), zz);
        w.right_trees.emplace_back(half+jx(rng), zz);
        w.end_z_m=zz;
    }
    return w;
}
static void blit_quad(cv::Mat& rgb, cv::Mat& depth, const std::vector<cv::Point>& poly, float z, const cv::Scalar& color) {
    if (poly.size()<3) return;
    cv::Rect bound=cv::boundingRect(poly) & cv::Rect(0,0,rgb.cols,rgb.rows);
    if (bound.width<=0) return;
    cv::Mat mask=cv::Mat::zeros(bound.size(), CV_8UC1);
    std::vector<cv::Point> local=poly;
    for (auto& p: local) { p.x-=bound.x; p.y-=bound.y; }
    cv::fillConvexPoly(mask, local, 255);
    for (int r=0;r<bound.height;++r) {
        const uint8_t* m=mask.ptr<uint8_t>(r);
        auto* pix=rgb.ptr<cv::Vec3b>(bound.y+r);
        float* d=depth.ptr<float>(bound.y+r);
        for (int c=0;c<bound.width;++c) if (m[c] && (d[bound.x+c]==0.f || z<d[bound.x+c])) {
            pix[bound.x+c]=cv::Vec3b((uchar)color[0],(uchar)color[1],(uchar)color[2]);
            d[bound.x+c]=z;
        }
    }
}
Frame render_frame(const OrchardWorld& world, const TractorPose& pose, const Config& cfg, double timestamp_s) {
    Frame frame; frame.K=Intrinsics::from_fov(cfg.image_width,cfg.image_height,cfg.depth_hfov_deg,cfg.depth_vfov_deg);
    frame.timestamp_s=timestamp_s;
    frame.rgb=cv::Mat(cfg.image_height,cfg.image_width,CV_8UC3, cv::Scalar(90,140,70));
    frame.depth_m=cv::Mat::zeros(cfg.image_height,cfg.image_width,CV_32F);
    const auto R=rotation_world_to_camera(pose.yaw_rad,cfg.pitch_rad());
    const float ox=pose.x_m, oy=cfg.camera_height_m, oz=pose.s_m;
    const Intrinsics& K=frame.K;
    auto draw_trunk=[&](float xw, float zw){
        float ys[4]={0.05f,0.05f,2.1f,2.1f}, xs[4]={xw-0.12f,xw+0.12f,xw+0.12f,xw-0.12f};
        std::vector<cv::Point> poly; float zsum=0; int n=0;
        for (int i=0;i<4;++i) {
            float cx,cy,cz; mul_R_vec(R, xs[i]-ox, ys[i]-oy, zw-oz, cx,cy,cz);
            if (cz<0.4f) return;
            float u,v; if (!K.cam_to_pixel(cx,cy,cz,u,v)) return;
            poly.emplace_back((int)std::lround(u),(int)std::lround(v)); zsum+=cz; n++;
        }
        blit_quad(frame.rgb, frame.depth_m, poly, zsum/n, cv::Scalar(28,62,96));
    };
    for (auto t: world.left_trees) draw_trunk(t.x, t.y);
    for (auto t: world.right_trees) draw_trunk(t.x, t.y);
    return frame;
}
}
