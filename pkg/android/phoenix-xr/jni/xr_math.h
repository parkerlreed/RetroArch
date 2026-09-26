// Minimal vector / quaternion / matrix helpers for the XR renderer.
// Matrices are column-major to match OpenGL.
#pragma once

#include <cmath>
#include <openxr/openxr.h>

struct Vec3 {
  float x = 0, y = 0, z = 0;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(Vec3 a) {
  float l = length(a);
  return l > 1e-6f ? a * (1.0f / l) : Vec3{0, 0, 0};
}

struct Quat {
  float x = 0, y = 0, z = 0, w = 1;
};

inline Quat operator*(Quat a, Quat b) {
  return {
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}
inline Quat conjugate(Quat q) { return {-q.x, -q.y, -q.z, q.w}; }
inline Quat normalize(Quat q) {
  float l = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (l < 1e-6f) return {};
  return {q.x / l, q.y / l, q.z / l, q.w / l};
}
inline Quat axisAngle(Vec3 axis, float radians) {
  axis = normalize(axis);
  float s = std::sin(radians * 0.5f);
  return {axis.x * s, axis.y * s, axis.z * s, std::cos(radians * 0.5f)};
}
inline Vec3 rotate(Quat q, Vec3 v) {
  Vec3 u{q.x, q.y, q.z};
  Vec3 t = cross(u, v) * 2.0f;
  return v + t * q.w + cross(u, t);
}
// Yaw (rotation about +Y) of the orientation's forward (-Z) direction projected on the floor.
inline float yawOf(Quat q) {
  Vec3 f = rotate(q, {0, 0, -1});
  return std::atan2(-f.x, -f.z);
}

struct Pose {
  Quat q;
  Vec3 p;
};

inline Vec3 transformPoint(const Pose& a, Vec3 v) { return rotate(a.q, v) + a.p; }
inline Pose operator*(const Pose& a, const Pose& b) {
  return {normalize(a.q * b.q), rotate(a.q, b.p) + a.p};
}
inline Pose inverse(const Pose& a) {
  Quat qi = conjugate(a.q);
  return {qi, rotate(qi, -a.p)};
}

inline Pose fromXr(const XrPosef& p) {
  return {{p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w},
          {p.position.x, p.position.y, p.position.z}};
}
inline XrPosef toXr(const Pose& p) {
  XrPosef r;
  r.orientation = {p.q.x, p.q.y, p.q.z, p.q.w};
  r.position = {p.p.x, p.p.y, p.p.z};
  return r;
}

struct Mat4 {
  float m[16];

  static Mat4 identity() {
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
  }

  // Rigid transform with uniform scale.
  static Mat4 fromPose(const Pose& pose, float scale = 1.0f) {
    const Quat& q = pose.q;
    float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    Mat4 r{};
    r.m[0] = (1 - 2 * (yy + zz)) * scale;
    r.m[1] = 2 * (xy + wz) * scale;
    r.m[2] = 2 * (xz - wy) * scale;
    r.m[4] = 2 * (xy - wz) * scale;
    r.m[5] = (1 - 2 * (xx + zz)) * scale;
    r.m[6] = 2 * (yz + wx) * scale;
    r.m[8] = 2 * (xz + wy) * scale;
    r.m[9] = 2 * (yz - wx) * scale;
    r.m[10] = (1 - 2 * (xx + yy)) * scale;
    r.m[12] = pose.p.x;
    r.m[13] = pose.p.y;
    r.m[14] = pose.p.z;
    r.m[15] = 1.0f;
    return r;
  }

  // Asymmetric OpenGL projection from an OpenXR field of view.
  static Mat4 projection(const XrFovf& fov, float nearZ, float farZ) {
    float l = std::tan(fov.angleLeft), r = std::tan(fov.angleRight);
    float u = std::tan(fov.angleUp), d = std::tan(fov.angleDown);
    float w = r - l, h = u - d;
    Mat4 m{};
    m.m[0] = 2.0f / w;
    m.m[5] = 2.0f / h;
    m.m[8] = (r + l) / w;
    m.m[9] = (u + d) / h;
    m.m[10] = -(farZ + nearZ) / (farZ - nearZ);
    m.m[11] = -1.0f;
    m.m[14] = -(2.0f * farZ * nearZ) / (farZ - nearZ);
    return m;
  }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (int c = 0; c < 4; ++c)
    for (int row = 0; row < 4; ++row) {
      float s = 0;
      for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
      r.m[c * 4 + row] = s;
    }
  return r;
}
