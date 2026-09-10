#include "orchard/perception.hpp"
#include "orchard/geometry.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
namespace orchard {
namespace {
bool fit_line_xz(const std::vector<float>& xs, const std::vector<float>& zs, float thresh, LineXZ& out) {
    const int n0 = (int)xs.size(); if (n0 < 2) return false;
    std::vector<char> mask(n0, 1); float intercept=0, slope=0;
    for (int iter=0; iter<4; ++iter) {
        double n=0,sumz=0,sumz2=0,sumx=0,sumxz=0;
        for (int i=0;i<n0;++i) if (mask[i]) { n++; sumz+=zs[i]; sumz2+=zs[i]*zs[i]; sumx+=xs[i]; sumxz+=xs[i]*zs[i]; }
        if (n<2) return false;
        double det=n*sumz2-sumz*sumz; if (std::abs(det)<1e-9) return false;
        intercept=(float)((sumz2*sumx-sumz*sumxz)/det); slope=(float)((n*sumxz-sumz*sumx)/det);
        for (int i=0;i<n0;++i) mask[i]=std::abs(intercept+slope*zs[i]-xs[i])<thresh;
    }
    int kept=0; for (char m: mask) kept+=m?1:0; if (kept<2) return false;
    out.intercept=intercept; out.slope=slope; return true;
}
struct Cluster { float x,z; int n; };
std::vector<Cluster> cluster_xz(const std::vector<float>& xs, const std::vector<float>& zs, float eps, int min_pts) {
    const int n=(int)xs.size(); std::vector<Cluster> out; if(!n) return out;
    std::vector<int> order(n); for(int i=0;i<n;++i) order[i]=i;
    std::sort(order.begin(), order.end(), [&](int a,int b){ return zs[a]<zs[b]; });
    std::vector<char> used(n,0); float eps2=eps*eps;
    for (int oi: order) {
        if (used[oi]) continue;
        float sx=0,sz=0; int count=0; std::vector<int> members;
        for (int j=0;j<n;++j) { float dx=xs[j]-xs[oi], dz=zs[j]-zs[oi]; if (dx*dx+dz*dz<eps2) { members.push_back(j); sx+=xs[j]; sz+=zs[j]; count++; } }
        if (count<min_pts) continue;
        for (int j: members) used[j]=1;
        out.push_back({sx/count, sz/count, count});
    }
    return out;
}
void inner_envelope(const std::vector<float>& xs, const std::vector<float>& zs, bool left,
                    std::vector<float>& ox, std::vector<float>& oz, float z_bin=0.9f, int min_pts=8) {
    ox.clear(); oz.clear(); if (xs.empty()) return;
    float z0=zs[0], z1=zs[0]; for (float z: zs) { z0=std::min(z0,z); z1=std::max(z1,z); }
    for (float b0=z0; b0<z1; b0+=z_bin) {
        float b1=b0+z_bin, best=left?-1e9f:1e9f; int count=0;
        for (size_t i=0;i<xs.size();++i) if (zs[i]>=b0 && zs[i]<b1) { count++; best=left?std::max(best,xs[i]):std::min(best,xs[i]); }
        if (count<min_pts) continue; ox.push_back(best); oz.push_back(0.5f*(b0+b1));
    }
}
struct PenCluster { float x,y,z; int n; float xz_span, y_span; };
std::vector<PenCluster> cluster_pens(const std::vector<float>& xs, const std::vector<float>& ys,
                                    const std::vector<float>& zs, float eps, int min_pts) {
    const int n=(int)xs.size(); std::vector<PenCluster> out; if(!n) return out;
    std::vector<int> order(n); for(int i=0;i<n;++i) order[i]=i;
    std::sort(order.begin(), order.end(), [&](int a,int b){ return zs[a]<zs[b]; });
    std::vector<char> used(n,0); float eps2=eps*eps;
    for (int oi: order) {
        if (used[oi]) continue;
        float sx=0,sy=0,sz=0,xmin=1e9f,xmax=-1e9f,ymin=1e9f,ymax=-1e9f,zmin=1e9f,zmax=-1e9f; int count=0;
        std::vector<int> members;
        for (int j=0;j<n;++j) {
            float dx=xs[j]-xs[oi], dz=zs[j]-zs[oi];
            if (dx*dx+dz*dz<eps2) {
                members.push_back(j); sx+=xs[j]; sy+=ys[j]; sz+=zs[j]; count++;
                xmin=std::min(xmin,xs[j]); xmax=std::max(xmax,xs[j]);
                ymin=std::min(ymin,ys[j]); ymax=std::max(ymax,ys[j]);
                zmin=std::min(zmin,zs[j]); zmax=std::max(zmax,zs[j]);
            }
        }
        if (count<min_pts) continue;
        for (int j: members) used[j]=1;
        float span=std::hypot(xmax-xmin, zmax-zmin);
        out.push_back({sx/count, sy/count, sz/count, count, span, ymax-ymin});
    }
    return out;
}
RowPerception perceive_pens(const Frame& frame, const Config& cfg) {
    RowPerception result;
    if (frame.depth_m.empty()) { result.notes.emplace_back("no_depth"); return result; }
    const cv::Mat& depth=frame.depth_m; const Intrinsics& K=frame.K;
    const int stride=std::max(1, cfg.pixel_stride);
    std::vector<float> xs,ys,zs;
    for (int v=0; v<depth.rows; v+=stride) {
        const float* row=depth.ptr<float>(v);
        for (int u=0; u<depth.cols; u+=stride) {
            float z=row[u]; if (z<cfg.desk_depth_min_m || z>cfg.desk_depth_max_m) continue;
            float x,y; K.pixel_to_cam((float)u,(float)v,z,x,y);
            if (std::abs(y)>cfg.desk_y_band_m) continue;
            xs.push_back(x); ys.push_back(y); zs.push_back(z);
        }
    }
    if ((int)xs.size()<20) { result.notes.emplace_back("no_pen_band"); return result; }
    std::vector<PenCluster> left, right;
    for (const auto& c: cluster_pens(xs,ys,zs,cfg.desk_cluster_eps_m,cfg.desk_cluster_min_points)) {
        if (c.xz_span>cfg.desk_max_span_m) continue;
        if (c.y_span<cfg.desk_min_vertical_m && c.n<cfg.desk_cluster_min_points*3) continue;
        if (c.x<-0.02f) left.push_back(c);
        else if (c.x>0.02f) right.push_back(c);
    }
    std::sort(left.begin(), left.end(), [](const PenCluster& a, const PenCluster& b){ return a.z<b.z; });
    std::sort(right.begin(), right.end(), [](const PenCluster& a, const PenCluster& b){ return a.z<b.z; });
    if (left.empty()) result.notes.emplace_back("left_pen_missing");
    if (right.empty()) result.notes.emplace_back("right_pen_missing");
    if (left.empty() || right.empty()) { result.notes.emplace_back("need_both_pens"); return result; }
    const PenCluster& L=left[0]; const PenCluster& R=right[0];
    const float gap=R.x-L.x;
    if (gap<cfg.desk_min_gap_m || gap>cfg.desk_max_gap_m) { result.notes.emplace_back("pen_gap_rejected"); return result; }
    const float center=0.5f*(L.x+R.x); const float lat=-center;
    if (std::abs(lat)>cfg.desk_max_lat_m) { result.notes.emplace_back("lat_rejected"); return result; }
    result.left_line=LineXZ{L.x,0.f}; result.right_line=LineXZ{R.x,0.f}; result.centerline=LineXZ{center,0.f};
    result.lateral_error_m=lat; result.heading_error_rad=0.f; result.row_width_m=gap;
    result.trunks.push_back({Trunk::Left,L.x,L.z,L.n});
    result.trunks.push_back({Trunk::Right,R.x,R.z,R.n});
    result.confidence=std::abs(lat)<0.15f?0.85f:0.70f;
    result.notes.emplace_back("desk_pens");
    return result;
}
}  // namespace
RowPerception perceive(const Frame& frame, const Config& cfg) {
    if (cfg.desk_mode) return perceive_pens(frame, cfg);
    RowPerception result;
    if (frame.depth_m.empty()) { result.notes.emplace_back("no_depth"); return result; }
    const cv::Mat& depth=frame.depth_m; const Intrinsics& K=frame.K;
    const int stride=std::max(1, cfg.pixel_stride);
    std::vector<float> gx, gz; int valid_n=0;
    for (int v=0; v<depth.rows; v+=stride) {
        const float* row=depth.ptr<float>(v);
        for (int u=0; u<depth.cols; u+=stride) {
            float z=row[u]; if (z<cfg.depth_min_m || z>cfg.depth_max_m) continue; ++valid_n;
            float x,y; K.pixel_to_cam((float)u,(float)v,z,x,y);
            float xr,agl,zf; camera_point_to_ground(x,y,z,cfg.pitch_rad(),cfg.camera_height_m,xr,agl,zf);
            if (agl<cfg.trunk_height_min_m || agl>cfg.trunk_height_max_m) continue;
            if (zf<cfg.depth_min_m || zf>cfg.depth_max_m) continue;
            gx.push_back(xr); gz.push_back(zf);
        }
    }
    if (valid_n<80) { result.notes.emplace_back("too_few_depth_pixels"); return result; }
    std::vector<float> lx,lz,rx,rz;
    for (size_t i=0;i<gx.size();++i) {
        if (gx[i]<-cfg.center_keepout_m) { lx.push_back(gx[i]); lz.push_back(gz[i]); }
        else if (gx[i]>cfg.center_keepout_m) { rx.push_back(gx[i]); rz.push_back(gz[i]); }
    }
    std::vector<float> lex,lez,rex,rez;
    inner_envelope(lx,lz,true,lex,lez); inner_envelope(rx,rz,false,rex,rez);
    LineXZ left,right; bool have_l=false, have_r=false;
    if ((int)lex.size()>=cfg.min_trunks_for_line) have_l=fit_line_xz(lex,lez,std::max(cfg.ransac_thresh_m,0.55f),left);
    else if (lx.size()>30) have_l=fit_line_xz(lx,lz,cfg.ransac_thresh_m,left);
    if ((int)rex.size()>=cfg.min_trunks_for_line) have_r=fit_line_xz(rex,rez,std::max(cfg.ransac_thresh_m,0.55f),right);
    else if (rx.size()>30) have_r=fit_line_xz(rx,rz,cfg.ransac_thresh_m,right);
    if (have_l) result.left_line=left; if (have_r) result.right_line=right;
    auto trunks_along=[&](const LineXZ& line, const std::vector<float>& xs, const std::vector<float>& zs, Trunk::Side side){
        std::vector<float> nx,nz;
        for (size_t i=0;i<xs.size();++i) if (std::abs(line.x_at(zs[i])-xs[i])<0.50f) { nx.push_back(xs[i]); nz.push_back(zs[i]); }
        for (const auto& c: cluster_xz(nx,nz,cfg.cluster_eps_m,cfg.cluster_min_points))
            result.trunks.push_back({side,c.x,c.z,c.n});
    };
    if (have_l) trunks_along(left,lx,lz,Trunk::Left);
    if (have_r) trunks_along(right,rx,rz,Trunk::Right);
    if (have_l && have_r) {
        LineXZ c; c.intercept=0.5f*(left.intercept+right.intercept); c.slope=0.5f*(left.slope+right.slope);
        result.centerline=c; result.row_width_m=std::abs(right.x_at(6.f)-left.x_at(6.f));
        result.lateral_error_m=-c.intercept; result.heading_error_rad=-c.heading_rad();
    } else if (have_l) {
        LineXZ c; c.intercept=left.intercept+0.5f*cfg.row_width_m; c.slope=left.slope;
        result.centerline=c; result.lateral_error_m=-c.intercept; result.heading_error_rad=-c.heading_rad();
    } else if (have_r) {
        LineXZ c; c.intercept=right.intercept-0.5f*cfg.row_width_m; c.slope=right.slope;
        result.centerline=c; result.lateral_error_m=-c.intercept; result.heading_error_rad=-c.heading_rad();
    }
    float conf=0; if (result.centerline) conf+=0.45f;
    if (result.trunks.size()>=4) conf+=0.35f; else if (result.trunks.size()>=2) conf+=0.12f;
    if (have_l && have_r) conf+=0.20f;
    result.confidence=std::min(1.f,conf);
    return result;
}
}
