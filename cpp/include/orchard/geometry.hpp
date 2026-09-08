#pragma once
#include <array>
#include <cmath>
namespace orchard {
inline std::array<float, 9> rotation_world_to_camera(float yaw_rad, float pitch_aero_rad) {
    const float cy = std::cos(yaw_rad), sy = std::sin(yaw_rad);
    const float look_down = -pitch_aero_rad;
    const float cp = std::cos(look_down), sp = std::sin(look_down);
    const float q00 = cy, q02 = -sy, q10 = -sp * sy, q11 = cp, q12 = -sp * cy, q20 = cp * sy, q21 = sp, q22 = cp * cy;
    return {q00, 0.f, q02, -q10, -q11, -q12, q20, q21, q22};
}
inline void mul_R_vec(const std::array<float, 9>& R, float x, float y, float z, float& ox, float& oy, float& oz) {
    ox = R[0]*x + R[1]*y + R[2]*z; oy = R[3]*x + R[4]*y + R[5]*z; oz = R[6]*x + R[7]*y + R[8]*z;
}
inline void camera_point_to_ground(float x, float y_down, float z, float pitch, float h, float& xr, float& agl, float& zf) {
    const auto R = rotation_world_to_camera(0.f, pitch);
    xr = R[0]*x + R[3]*y_down + R[6]*z;
    agl = h + (R[1]*x + R[4]*y_down + R[7]*z);
    zf = R[2]*x + R[5]*y_down + R[8]*z;
}
inline void ground_to_camera(float xr, float agl, float zf, float pitch, float h, float& x, float& y, float& z) {
    mul_R_vec(rotation_world_to_camera(0.f, pitch), xr, agl - h, zf, x, y, z);
}
}
