#include "crt_scene.h"

#include "tv_model.h"
#include "xr_actions.h"

#include <GLES2/gl2ext.h>
#include <android/log.h>

#include <algorithm>
#include <vector>

#define LOG_TAG "RetroArch-XR"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// Material colours are linear; the swapchain is sRGB so the GPU encodes on write.
constexpr Vec3 kPlastic{0.045f, 0.044f, 0.048f};
constexpr Vec3 kTrim{0.018f, 0.018f, 0.020f};
constexpr Vec3 kLedOn{0.05f, 0.9f, 0.12f};
constexpr Vec3 kLedIdle{0.9f, 0.25f, 0.02f};
constexpr Vec3 kController{0.3f, 0.3f, 0.32f};
constexpr Vec3 kRay{0.55f, 0.75f, 1.0f};
constexpr Vec3 kRayHit{1.0f, 0.85f, 0.35f};
constexpr Vec3 kSlotInterior{0.003f, 0.003f, 0.0035f};
constexpr Vec3 kAccent{0.30f, 0.30f, 0.32f};
constexpr Vec3 kButton{0.10f, 0.10f, 0.11f};
constexpr Vec3 kButtonHover{0.26f, 0.26f, 0.30f};
constexpr Vec3 kIcon{0.75f, 0.75f, 0.78f};

// Front-panel layout. Row 1 sits under the tube: cassette slot on the left, transport buttons on
// the right. Row 2 has the speaker grille, volume buttons and the power LED.
constexpr float kRow1Y = -0.2075f;
constexpr float kRow2Y = -0.262f;
constexpr float kSlotX0 = -0.21f, kSlotX1 = -0.01f, kSlotY0 = -0.225f, kSlotY1 = -0.19f;
constexpr float kSlotDepth = -0.035f;
constexpr float kButtonFrontZ = 0.008f, kButtonBackZ = -0.004f, kButtonTravel = 0.005f;

enum Icon { kIconEject, kIconRewind, kIconPlayPause, kIconFastForward, kIconStop, kIconMinus, kIconPlus };

const CrtButton kButtons[] = {
    {kExit, 0.032f, kRow1Y, 0.017f, 0.011f, kIconEject, false},
    {kSeekBack, 0.072f, kRow1Y, 0.017f, 0.011f, kIconRewind, true},
    {kTogglePause, 0.112f, kRow1Y, 0.017f, 0.011f, kIconPlayPause, false},
    {kSeekForward, 0.152f, kRow1Y, 0.017f, 0.011f, kIconFastForward, true},
    {kStop, 0.192f, kRow1Y, 0.017f, 0.011f, kIconStop, false},
    {kVolumeDown, 0.11f, kRow2Y, 0.015f, 0.009f, kIconMinus, true},
    {kVolumeUp, 0.15f, kRow2Y, 0.015f, 0.009f, kIconPlus, true},
};
constexpr int kButtonCount = sizeof(kButtons) / sizeof(kButtons[0]);
static_assert(kButtonCount <= CrtScene::kMaxButtons, "raise kMaxButtons");

// Tube curvature: how far the centre of the glass bulges out from its edges, and where the edges
// sit behind the bezel face.
constexpr float kScreenEdgeZ = -0.02f;
constexpr float kScreenBulge = 0.014f;
// Effective scanline count across the tube height.
constexpr float kScanlines = 240.0f;

const char* kLitVs = R"(#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec2 aUv;
uniform mat4 uModel;
uniform mat4 uViewProj;
out vec3 vWorld;
out vec3 vNrm;
out vec2 vUv;
void main() {
  vec4 w = uModel * vec4(aPos, 1.0);
  vWorld = w.xyz;
  vNrm = mat3(uModel) * aNrm;
  vUv = aUv;
  gl_Position = uViewProj * w;
}
)";

const char* kLitFs = R"(#version 300 es
precision mediump float;
in vec3 vWorld;
in vec3 vNrm;
in vec2 vUv;
uniform vec3 uColor;
uniform vec3 uEye;
uniform float uEmissive;
uniform float uSpec;
out vec4 fragColor;
void main() {
  vec3 n = normalize(vNrm);
  if (!gl_FrontFacing) n = -n;
  vec3 l = normalize(vec3(0.3, 1.0, 0.5));
  vec3 v = normalize(uEye - vWorld);
  vec3 h = normalize(l + v);
  float diffuse = max(dot(n, l), 0.0);
  float sky = 0.5 + 0.5 * n.y;
  float spec = pow(max(dot(n, h), 0.0), 40.0) * uSpec;
  vec3 c = uColor * (0.35 * sky + 0.9 * diffuse) + vec3(spec);
  fragColor = vec4(mix(c, uColor, uEmissive), 1.0);
}
)";

const char* kScreenFs = R"(#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
in vec3 vWorld;
in vec3 vNrm;
in vec2 vUv;
uniform samplerExternalOES uTex;
uniform mat4 uTexMatrix;
uniform float uHasFrame;
uniform vec3 uEye;
uniform float uLines;
uniform float uScanlineStrength;
uniform float uScanlineFade;
uniform float uReflections;
// Glass extent in its UVs (min.xy, max.xy); the video covers all of it.
uniform vec4 uVideoRect;
uniform vec2 uVideoFlip;
uniform float uUseMask;
uniform sampler2D uMask;
uniform float uMaskThreshold;
uniform sampler2D uOcclusion;
uniform float uOcclusionStrength;
uniform float uVignette;
out vec4 fragColor;
void main() {
  vec3 n = normalize(vNrm);
  if (!gl_FrontFacing) n = -n;
  vec3 v = normalize(uEye - vWorld);

  vec2 pic = (vUv - uVideoRect.xy) / (uVideoRect.zw - uVideoRect.xy);
  pic = mix(pic, 1.0 - pic, uVideoFlip);
  vec2 tc = (uTexMatrix * vec4(clamp(pic, 0.0, 1.0), 0.0, 1.0)).xy;
  // mpv writes display-referred sRGB; decode so the sRGB swapchain round-trips it unchanged.
  vec3 video = pow(texture(uTex, tc).rgb, vec3(2.2)) * uHasFrame;

  // Scanlines, faded out once they get close to pixel size to avoid moire.
  float phase = pic.y * uLines;
  float fw = fwidth(phase);
  float fade = mix(1.0, clamp(1.0 - (fw - 0.2) * 2.5, 0.0, 1.0), uScanlineFade);
  float strength = uScanlineStrength * fade;
  float line = 0.5 + 0.5 * cos(phase * 6.2831853);
  video *= mix(1.0, (0.55 + 0.45 * line) * 1.2, strength);

  // Tube falloff towards the edges: the model's own occlusion if it has one, otherwise a vignette.
  vec2 d = pic * 2.0 - 1.0;
  video *= 1.0 - uVignette * dot(d * d, vec2(1.0));
  float ao = mix(1.0, texture(uOcclusion, vUv).r, uOcclusionStrength);
  video *= ao;

  // Dark glass with a soft fresnel sheen and a broad overhead highlight.
  vec3 glass = vec3(0.0015, 0.0018, 0.0016);
  float fresnel = pow(1.0 - max(dot(n, v), 0.0), 4.0);
  vec3 l = normalize(vec3(0.3, 1.0, 0.5));
  float highlight = pow(max(dot(reflect(-l, n), v), 0.0), 60.0);
  // Keep the model's painted mask (with its rounded corners) over the edges of the picture.
  float inside = 1.0;
  vec3 maskColor = vec3(0.0);
  if (uUseMask > 0.5) {
    maskColor = texture(uMask, vUv).rgb * ao;
    inside = smoothstep(uMaskThreshold * 0.7, uMaskThreshold * 1.3, dot(maskColor, vec3(1.0 / 3.0)));
  }
  vec3 c = mix(maskColor, glass + video, inside) +
           (vec3(0.02) * fresnel + vec3(0.05) * highlight) * uReflections;
  fragColor = vec4(c, 1.0);
}
)";

const char* kPbrVs = R"(#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec2 aUv;
layout(location = 3) in vec4 aTan;
uniform mat4 uModel;
uniform mat4 uViewProj;
out vec3 vWorld;
out vec3 vNrm;
out vec2 vUv;
out vec4 vTan;
void main() {
  vec4 w = uModel * vec4(aPos, 1.0);
  vWorld = w.xyz;
  vNrm = mat3(uModel) * aNrm;
  vTan = vec4(mat3(uModel) * aTan.xyz, aTan.w);
  vUv = aUv;
  gl_Position = uViewProj * w;
}
)";

// Metallic-roughness PBR with a key light, a fill light and a soft "room" environment standing
// in for the passthrough surroundings.
const char* kPbrFs = R"(#version 300 es
precision highp float;
in vec3 vWorld;
in vec3 vNrm;
in vec2 vUv;
in vec4 vTan;
uniform sampler2D uBase;
uniform sampler2D uMetalRough;
uniform sampler2D uNormalMap;
uniform sampler2D uOcclusionMap;
uniform vec4 uBaseFactor;
uniform float uMetallic;
uniform float uRoughness;
uniform float uNormalScale;
uniform float uUseNormalMap;
uniform float uOcclusionStrength;
uniform float uHighlight;
uniform vec3 uEye;
out vec4 fragColor;

const float PI = 3.14159265;

vec3 environment(vec3 dir, float roughness) {
  // Bright ceiling, mid walls, dark floor, blurred more as roughness rises.
  float y = dir.y;
  vec3 sky = vec3(0.55, 0.56, 0.58);
  vec3 wall = vec3(0.22, 0.21, 0.20);
  vec3 floorC = vec3(0.06, 0.055, 0.05);
  float soft = mix(0.15, 0.9, roughness);
  vec3 c = mix(wall, sky, smoothstep(-soft * 0.3, soft + 0.2, y));
  c = mix(floorC, c, smoothstep(-0.6 - soft, -0.05, y));
  return c;
}

vec3 brdf(vec3 n, vec3 v, vec3 l, vec3 radiance, vec3 diffuse, vec3 f0, float rough) {
  vec3 h = normalize(l + v);
  float nl = max(dot(n, l), 0.0);
  float nv = max(dot(n, v), 1e-4);
  float nh = max(dot(n, h), 0.0);
  float vh = max(dot(v, h), 0.0);
  float a = rough * rough;
  float a2 = a * a;
  float dd = nh * nh * (a2 - 1.0) + 1.0;
  float D = a2 / (PI * dd * dd);
  float k = (rough + 1.0) * (rough + 1.0) / 8.0;
  float G = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
  vec3 F = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
  vec3 spec = D * G * F / (4.0 * nv * nl + 1e-4);
  vec3 kd = (1.0 - F);
  return (kd * diffuse / PI + spec) * radiance * nl;
}

void main() {
  vec4 base = texture(uBase, vUv) * uBaseFactor;
  vec3 mr = texture(uMetalRough, vUv).rgb;
  float rough = clamp(uRoughness * mr.g, 0.04, 1.0);
  float metal = clamp(uMetallic * mr.b, 0.0, 1.0);
  float ao = mix(1.0, texture(uOcclusionMap, vUv).r, uOcclusionStrength);

  vec3 n = normalize(vNrm);
  if (!gl_FrontFacing) n = -n;
  if (uUseNormalMap > 0.5) {
    vec3 t = normalize(vTan.xyz - n * dot(n, vTan.xyz));
    vec3 b = cross(n, t) * vTan.w;
    vec3 tn = texture(uNormalMap, vUv).xyz * 2.0 - 1.0;
    tn.xy *= uNormalScale;
    n = normalize(mat3(t, b, n) * tn);
  }
  vec3 v = normalize(uEye - vWorld);
  vec3 f0 = mix(vec3(0.04), base.rgb, metal);
  vec3 diffuse = base.rgb * (1.0 - metal);

  vec3 c = brdf(n, v, normalize(vec3(0.35, 1.0, 0.6)), vec3(2.6), diffuse, f0, rough);
  c += brdf(n, v, normalize(vec3(-0.7, 0.3, 0.4)), vec3(0.7), diffuse, f0, rough);

  float nv = max(dot(n, v), 0.0);
  vec3 fEnv = f0 + (max(vec3(1.0 - rough), f0) - f0) * pow(1.0 - nv, 5.0);
  vec3 irradiance = environment(n, 1.0);
  vec3 reflection = environment(reflect(-v, n), rough);
  c += (diffuse * irradiance * (1.0 - fEnv) + reflection * fEnv * (1.0 - 0.5 * rough)) * ao;

  c += uHighlight * vec3(0.10, 0.10, 0.13);
  fragColor = vec4(c, 1.0);
}
)";

const char* kOccluderVs = R"(#version 300 es
layout(location = 0) in vec3 aPos;
uniform mat4 uViewProj;
uniform vec4 uSpheres[128];
void main() {
  vec4 s = uSpheres[gl_InstanceID];
  gl_Position = uViewProj * vec4(s.xyz + aPos * s.w, 1.0);
}
)";

const char* kHandVs = R"(#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec4 aJoints;
layout(location = 3) in vec4 aWeights;
uniform mat4 uViewProj;
uniform mat4 uSkin[26];
uniform float uInflate;
void main() {
  ivec4 j = clamp(ivec4(aJoints), 0, 25);
  mat4 m = uSkin[j.x] * aWeights.x + uSkin[j.y] * aWeights.y + uSkin[j.z] * aWeights.z +
           uSkin[j.w] * aWeights.w;
  vec3 p = (m * vec4(aPos, 1.0)).xyz;
  vec3 n = normalize(mat3(m) * aNrm);
  // Grow the hand a little so small tracking errors don't show slivers of the set over it.
  gl_Position = uViewProj * vec4(p + n * uInflate, 1.0);
}
)";

// Transparent black: the compositor shows passthrough here.
const char* kOccluderFs = R"(#version 300 es
precision mediump float;
out vec4 fragColor;
void main() { fragColor = vec4(0.0); }
)";

GLuint compile(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(s, sizeof(log), nullptr, log);
    LOGE("shader compile failed: %s", log);
    glDeleteShader(s);
    return 0;
  }
  return s;
}

GLuint link(const char* vs, const char* fs) {
  GLuint v = compile(GL_VERTEX_SHADER, vs);
  GLuint f = compile(GL_FRAGMENT_SHADER, fs);
  if (!v || !f) return 0;
  GLuint p = glCreateProgram();
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  glDeleteShader(v);
  glDeleteShader(f);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetProgramInfoLog(p, sizeof(log), nullptr, log);
    LOGE("program link failed: %s", log);
    glDeleteProgram(p);
    return 0;
  }
  return p;
}

struct MeshBuilder {
  std::vector<float> verts;  // pos3 nrm3 uv2
  std::vector<GLuint> indices;

  GLuint vertex(Vec3 p, Vec3 n, float u = 0, float v = 0) {
    verts.insert(verts.end(), {p.x, p.y, p.z, n.x, n.y, n.z, u, v});
    return static_cast<GLuint>(verts.size() / 8 - 1);
  }

  void tri(Vec3 a, Vec3 b, Vec3 c) {
    Vec3 n = normalize(cross(b - a, c - a));
    GLuint ia = vertex(a, n), ib = vertex(b, n), ic = vertex(c, n);
    indices.insert(indices.end(), {ia, ib, ic});
  }

  // Corners in order around the face; the normal follows the winding.
  void quad(Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
    Vec3 n = normalize(cross(b - a, d - a));
    GLuint ia = vertex(a, n), ib = vertex(b, n), ic = vertex(c, n), id = vertex(d, n);
    indices.insert(indices.end(), {ia, ib, ic, ia, ic, id});
  }

  // Flat rectangle on a constant-z plane.
  void rectZ(float x0, float y0, float x1, float y1, float z) {
    quad({x0, y0, z}, {x1, y0, z}, {x1, y1, z}, {x0, y1, z});
  }

  // Joins two axis-aligned rectangles at different depths with four side walls.
  void loft(float ax0, float ay0, float ax1, float ay1, float az, float bx0, float by0, float bx1,
            float by1, float bz) {
    Vec3 a[4] = {{ax0, ay0, az}, {ax1, ay0, az}, {ax1, ay1, az}, {ax0, ay1, az}};
    Vec3 b[4] = {{bx0, by0, bz}, {bx1, by0, bz}, {bx1, by1, bz}, {bx0, by1, bz}};
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) % 4;
      quad(a[j], a[i], b[i], b[j]);
    }
  }

  void box(Vec3 lo, Vec3 hi) {
    rectZ(lo.x, lo.y, hi.x, hi.y, hi.z);
    rectZ(lo.x, lo.y, hi.x, hi.y, lo.z);
    loft(lo.x, lo.y, hi.x, hi.y, hi.z, lo.x, lo.y, hi.x, hi.y, lo.z);
  }

  Mesh upload() const {
    Mesh m;
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ibo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(),
                 GL_STATIC_DRAW);
    const GLsizei stride = 8 * sizeof(float);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<const void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<const void*>(6 * sizeof(float)));
    glBindVertexArray(0);
    m.count = static_cast<GLsizei>(indices.size());
    return m;
  }
};

void freeMesh(Mesh& m) {
  if (m.vao) glDeleteVertexArrays(1, &m.vao);
  if (m.vbo) glDeleteBuffers(1, &m.vbo);
  if (m.ibo) glDeleteBuffers(1, &m.ibo);
  m = {};
}

Mesh buildBody() {
  constexpr float ox0 = CrtScene::kBoundsMin.x, ox1 = CrtScene::kBoundsMax.x;
  constexpr float oy0 = CrtScene::kBoundsMin.y, oy1 = 0.19f;
  constexpr float sx = CrtScene::kScreenHalfW, sy = CrtScene::kScreenHalfH;
  MeshBuilder b;
  // Bezel face around the tube opening.
  b.rectZ(ox0, sy, ox1, oy1, 0);
  // Control panel below the tube, with an opening for the cassette slot.
  b.rectZ(ox0, oy0, ox1, kSlotY0, 0);
  b.rectZ(ox0, kSlotY1, ox1, -sy, 0);
  b.rectZ(ox0, kSlotY0, kSlotX0, kSlotY1, 0);
  b.rectZ(kSlotX1, kSlotY0, ox1, kSlotY1, 0);
  b.rectZ(ox0, -sy, -sx, sy, 0);
  b.rectZ(sx, -sy, ox1, sy, 0);
  // Recess between the bezel and the edge of the glass.
  b.loft(-sx, -sy, sx, sy, 0, -sx, -sy, sx, sy, kScreenEdgeZ);
  // Front cabinet section, then the tapered tube housing, then the back cap.
  constexpr float frontDepth = -0.14f, backZ = CrtScene::kBoundsMin.z;
  b.loft(ox0, oy0, ox1, oy1, 0, ox0, oy0, ox1, oy1, frontDepth);
  b.loft(ox0, oy0, ox1, oy1, frontDepth, -0.16f, -0.26f, 0.16f, 0.10f, backZ);
  b.rectZ(-0.16f, -0.26f, 0.16f, 0.10f, backZ);
  return b.upload();
}

Mesh buildTrim() {
  MeshBuilder b;
  // Spring-loaded cassette door, set just inside the slot opening.
  b.rectZ(kSlotX0 + 0.0025f, kSlotY0 + 0.0025f, kSlotX1 - 0.0025f, kSlotY1 - 0.0025f, -0.006f);
  // Speaker grille slots on the lower left.
  for (int i = 0; i < 6; ++i) {
    float y = -0.285f + i * 0.008f;
    b.rectZ(-0.235f, y, -0.06f, y + 0.004f, 0.001f);
  }
  return b.upload();
}

Mesh buildSlot() {
  MeshBuilder b;
  b.loft(kSlotX0, kSlotY0, kSlotX1, kSlotY1, 0, kSlotX0, kSlotY0, kSlotX1, kSlotY1, kSlotDepth);
  b.rectZ(kSlotX0, kSlotY0, kSlotX1, kSlotY1, kSlotDepth);
  return b.upload();
}

Mesh buildAccent() {
  MeshBuilder b;
  // Chrome pinstripe between the tube and the VCR section.
  b.rectZ(-0.24f, -0.172f, 0.24f, -0.169f, 0.0008f);
  return b.upload();
}

Mesh buildLed() {
  MeshBuilder b;
  b.box({0.205f, kRow2Y - 0.004f, 0.0f}, {0.215f, kRow2Y + 0.004f, 0.004f});
  return b.upload();
}

// Button body in button-local space; the draw call translates it to its centre.
Mesh buildButtonBody(const CrtButton& button) {
  MeshBuilder b;
  b.box({-button.hw, -button.hh, kButtonBackZ}, {button.hw, button.hh, kButtonFrontZ});
  return b.upload();
}

Mesh buildButtonIcon(int icon) {
  constexpr float z = kButtonFrontZ + 0.0006f;
  MeshBuilder b;
  auto t = [&](float ax, float ay, float bx, float by, float cx, float cy) {
    b.tri({ax, ay, z}, {bx, by, z}, {cx, cy, z});
  };
  auto r = [&](float x0, float y0, float x1, float y1) { b.rectZ(x0, y0, x1, y1, z); };
  switch (icon) {
    case kIconEject:
      t(-0.006f, -0.0005f, 0.006f, -0.0005f, 0.0f, 0.0055f);
      r(-0.006f, -0.0055f, 0.006f, -0.003f);
      break;
    case kIconRewind:
      t(0.0f, -0.005f, 0.0f, 0.005f, -0.008f, 0.0f);
      t(0.008f, -0.005f, 0.008f, 0.005f, 0.0f, 0.0f);
      break;
    case kIconPlayPause:
      t(-0.010f, -0.005f, -0.002f, 0.0f, -0.010f, 0.005f);
      r(0.002f, -0.005f, 0.0045f, 0.005f);
      r(0.0065f, -0.005f, 0.009f, 0.005f);
      break;
    case kIconFastForward:
      t(-0.008f, -0.005f, 0.0f, 0.0f, -0.008f, 0.005f);
      t(0.0f, -0.005f, 0.008f, 0.0f, 0.0f, 0.005f);
      break;
    case kIconStop:
      r(-0.0045f, -0.0045f, 0.0045f, 0.0045f);
      break;
    case kIconPlus:
      r(-0.0012f, -0.005f, 0.0012f, 0.005f);
      [[fallthrough]];
    case kIconMinus:
      r(-0.005f, -0.0012f, 0.005f, 0.0012f);
      break;
  }
  return b.upload();
}

Mesh buildScreen() {
  constexpr int cols = 48, rows = 36;
  constexpr float sx = CrtScene::kScreenHalfW, sy = CrtScene::kScreenHalfH;
  MeshBuilder b;
  for (int r = 0; r <= rows; ++r) {
    for (int c = 0; c <= cols; ++c) {
      float u = static_cast<float>(c) / cols, v = static_cast<float>(r) / rows;
      float nx = u * 2 - 1, ny = v * 2 - 1;
      float fx = 1 - nx * nx, fy = 1 - ny * ny;
      float z = kScreenEdgeZ + kScreenBulge * fx * fy;
      // Surface gradient for the normal.
      float dzdx = kScreenBulge * (-2 * nx) * fy / sx;
      float dzdy = kScreenBulge * fx * (-2 * ny) / sy;
      b.vertex({nx * sx, ny * sy, z}, normalize(Vec3{-dzdx, -dzdy, 1}), u, v);
    }
  }
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      GLuint i0 = r * (cols + 1) + c, i1 = i0 + 1, i2 = i0 + cols + 1, i3 = i2 + 1;
      b.indices.insert(b.indices.end(), {i0, i1, i3, i0, i3, i2});
    }
  }
  return b.upload();
}

Mesh buildUnitSphere() {
  constexpr int rings = 8, segments = 12;
  MeshBuilder b;
  for (int r = 0; r <= rings; ++r) {
    float phi = 3.14159265f * r / rings;
    for (int s = 0; s <= segments; ++s) {
      float theta = 6.2831853f * s / segments;
      Vec3 p{std::sin(phi) * std::cos(theta), std::cos(phi), std::sin(phi) * std::sin(theta)};
      b.vertex(p, p);
    }
  }
  for (int r = 0; r < rings; ++r) {
    for (int s = 0; s < segments; ++s) {
      GLuint i0 = r * (segments + 1) + s, i1 = i0 + 1, i2 = i0 + segments + 1, i3 = i2 + 1;
      b.indices.insert(b.indices.end(), {i0, i2, i1, i1, i2, i3});
    }
  }
  return b.upload();
}

Mesh buildUnitBox() {
  MeshBuilder b;
  b.box({-0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, 0.5f});
  return b.upload();
}

// Non-uniform scale about the origin, applied before `base`.
Mat4 scaled(const Mat4& base, Vec3 s, Vec3 offset) {
  Mat4 local = Mat4::identity();
  local.m[0] = s.x;
  local.m[5] = s.y;
  local.m[10] = s.z;
  local.m[12] = offset.x;
  local.m[13] = offset.y;
  local.m[14] = offset.z;
  return base * local;
}

}  // namespace

bool CrtScene::init(const char* modelPath) {
  litProgram_ = link(kLitVs, kLitFs);
  screenProgram_ = link(kLitVs, kScreenFs);
  pbrProgram_ = link(kPbrVs, kPbrFs);
  occluderProgram_ = link(kOccluderVs, kOccluderFs);
  handProgram_ = link(kHandVs, kOccluderFs);
  if (!litProgram_ || !screenProgram_ || !pbrProgram_ || !occluderProgram_ || !handProgram_) {
    return false;
  }
  box_ = buildUnitBox();
  sphere_ = buildUnitSphere();

  if (modelPath && *modelPath) {
    model_ = new TvModel();
    if (model_->load(modelPath)) {
      boundsMin_ = model_->boundsMin();
      boundsMax_ = model_->boundsMax();
      screenHalfW_ = model_->screenHalfW();
      screenHalfH_ = model_->screenHalfH();
      buttons_ = model_->buttons();
      return true;
    }
    model_->destroy();
    delete model_;
    model_ = nullptr;
  }

  buttons_.assign(kButtons, kButtons + kButtonCount);
  body_ = buildBody();
  trim_ = buildTrim();
  slot_ = buildSlot();
  accent_ = buildAccent();
  for (int i = 0; i < kButtonCount; ++i) {
    buttonBodies_[i] = buildButtonBody(kButtons[i]);
    buttonIcons_[i] = buildButtonIcon(kButtons[i].icon);
  }
  led_ = buildLed();
  screen_ = buildScreen();
  return true;
}

float CrtScene::screenDiagonalInches() const {
  return std::sqrt(screenHalfW_ * screenHalfW_ + screenHalfH_ * screenHalfH_) * 2.0f / 0.0254f;
}

void CrtScene::destroy() {
  for (Mesh* m : {&body_, &trim_, &slot_, &accent_, &led_, &screen_, &box_, &sphere_,
                  &handMeshes_[0], &handMeshes_[1]}) {
    freeMesh(*m);
  }
  for (int i = 0; i < kMaxButtons; ++i) {
    freeMesh(buttonBodies_[i]);
    freeMesh(buttonIcons_[i]);
  }
  if (model_) {
    model_->destroy();
    delete model_;
    model_ = nullptr;
  }
  if (litProgram_) glDeleteProgram(litProgram_);
  if (screenProgram_) glDeleteProgram(screenProgram_);
  if (pbrProgram_) glDeleteProgram(pbrProgram_);
  if (occluderProgram_) glDeleteProgram(occluderProgram_);
  if (handProgram_) glDeleteProgram(handProgram_);
  litProgram_ = screenProgram_ = pbrProgram_ = occluderProgram_ = handProgram_ = 0;
}

void CrtScene::setVideo(GLuint externalTexture, const float* texMatrix, bool hasFrame) {
  videoTexture_ = externalTexture;
  if (texMatrix) std::copy(texMatrix, texMatrix + 16, texMatrix_);
  hasFrame_ = hasFrame;
}

void CrtScene::drawLit(const Mesh& mesh, const Mat4& model, const Mat4& viewProj, Vec3 eye,
                       Vec3 color, float emissive, float spec) {
  glUniformMatrix4fv(glGetUniformLocation(litProgram_, "uModel"), 1, GL_FALSE, model.m);
  glUniformMatrix4fv(glGetUniformLocation(litProgram_, "uViewProj"), 1, GL_FALSE, viewProj.m);
  glUniform3f(glGetUniformLocation(litProgram_, "uColor"), color.x, color.y, color.z);
  glUniform3f(glGetUniformLocation(litProgram_, "uEye"), eye.x, eye.y, eye.z);
  glUniform1f(glGetUniformLocation(litProgram_, "uEmissive"), emissive);
  glUniform1f(glGetUniformLocation(litProgram_, "uSpec"), spec);
  glBindVertexArray(mesh.vao);
  glDrawElements(GL_TRIANGLES, mesh.count, GL_UNSIGNED_INT, nullptr);
}

void CrtScene::draw(const Mat4& viewProj, Vec3 eye, const Pose& crtPose, float crtScale,
                    const ControllerVisual* controllers, int controllerCount) {
  Mat4 model = Mat4::fromPose(crtPose, crtScale);

  if (model_) {
    glUseProgram(pbrProgram_);
    glUniformMatrix4fv(glGetUniformLocation(pbrProgram_, "uViewProj"), 1, GL_FALSE, viewProj.m);
    glUniform3f(glGetUniformLocation(pbrProgram_, "uEye"), eye.x, eye.y, eye.z);
    model_->drawBody(pbrProgram_, model, hoveredButtons_, pressedButtons_);
  }

  glUseProgram(litProgram_);
  if (!model_) drawProcedural(model, viewProj, eye);

  for (int i = 0; i < controllerCount; ++i) {
    const ControllerVisual& c = controllers[i];
    if (!c.active) continue;
    Mat4 aim = Mat4::fromPose(c.aim);
    if (c.drawController) {
      drawLit(box_, scaled(aim, {0.025f, 0.025f, 0.08f}, {0, 0, 0.03f}), viewProj, eye,
              kController, 0.0f, 0.3f);
    }
    if (c.rayLength > 0) {
      drawLit(box_, scaled(aim, {0.002f, 0.002f, c.rayLength}, {0, 0, -c.rayLength * 0.5f}),
              viewProj, eye, c.highlighted ? kRayHit : kRay, 1.0f, 0.0f);
    }
  }

  glUseProgram(screenProgram_);
  glUniformMatrix4fv(glGetUniformLocation(screenProgram_, "uModel"), 1, GL_FALSE, model.m);
  glUniformMatrix4fv(glGetUniformLocation(screenProgram_, "uViewProj"), 1, GL_FALSE, viewProj.m);
  glUniformMatrix4fv(glGetUniformLocation(screenProgram_, "uTexMatrix"), 1, GL_FALSE, texMatrix_);
  glUniform3f(glGetUniformLocation(screenProgram_, "uEye"), eye.x, eye.y, eye.z);
  glUniform1f(glGetUniformLocation(screenProgram_, "uHasFrame"),
              hasFrame_ && videoTexture_ ? 1.0f : 0.0f);
  glUniform1f(glGetUniformLocation(screenProgram_, "uLines"), kScanlines);
  glUniform1f(glGetUniformLocation(screenProgram_, "uScanlineStrength"), scanlines_);
  glUniform1f(glGetUniformLocation(screenProgram_, "uScanlineFade"), scanlineFade_ ? 1.0f : 0.0f);
  glUniform1f(glGetUniformLocation(screenProgram_, "uReflections"), reflections_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_EXTERNAL_OES, videoTexture_);
  glUniform1i(glGetUniformLocation(screenProgram_, "uTex"), 0);
  const TvModel::ScreenMask noMask;
  const TvModel::ScreenMask& mask = model_ ? model_->screenMask() : noMask;
  glUniform4f(glGetUniformLocation(screenProgram_, "uVideoRect"), mask.uvMin[0], mask.uvMin[1],
              mask.uvMax[0], mask.uvMax[1]);
  glUniform2f(glGetUniformLocation(screenProgram_, "uVideoFlip"), mask.flipU ? 1.0f : 0.0f,
              mask.flipV ? 1.0f : 0.0f);
  glUniform1f(glGetUniformLocation(screenProgram_, "uUseMask"), mask.texture ? 1.0f : 0.0f);
  glUniform1f(glGetUniformLocation(screenProgram_, "uMaskThreshold"), mask.threshold);
  glUniform1f(glGetUniformLocation(screenProgram_, "uOcclusionStrength"),
              mask.occlusion ? mask.occlusionStrength : 0.0f);
  glUniform1f(glGetUniformLocation(screenProgram_, "uVignette"), model_ ? 0.0f : 0.22f);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, mask.texture);
  glUniform1i(glGetUniformLocation(screenProgram_, "uMask"), 1);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, mask.occlusion);
  glUniform1i(glGetUniformLocation(screenProgram_, "uOcclusion"), 2);
  glActiveTexture(GL_TEXTURE0);
  if (model_) {
    model_->drawScreen();
  } else {
    glBindVertexArray(screen_.vao);
    glDrawElements(GL_TRIANGLES, screen_.count, GL_UNSIGNED_INT, nullptr);
  }

  glBindVertexArray(0);
}

void CrtScene::drawProcedural(const Mat4& model, const Mat4& viewProj, Vec3 eye) {
  drawLit(body_, model, viewProj, eye, kPlastic, 0.0f, 0.25f);
  drawLit(trim_, model, viewProj, eye, kTrim, 0.0f, 0.1f);
  drawLit(slot_, model, viewProj, eye, kSlotInterior, 0.0f, 0.0f);
  drawLit(accent_, model, viewProj, eye, kAccent, 0.0f, 0.6f);
  for (int i = 0; i < kButtonCount; ++i) {
    const CrtButton& button = kButtons[i];
    bool hovered = hoveredButtons_ & (1u << i);
    bool pressed = pressedButtons_ & (1u << i);
    Mat4 at = scaled(model, {1, 1, 1}, {button.cx, button.cy, pressed ? -kButtonTravel : 0.0f});
    drawLit(buttonBodies_[i], at, viewProj, eye, hovered ? kButtonHover : kButton, 0.0f, 0.35f);
    drawLit(buttonIcons_[i], at, viewProj, eye, kIcon, hovered ? 0.6f : 0.25f, 0.0f);
  }
  drawLit(led_, model, viewProj, eye, playing_ ? kLedOn : kLedIdle, 1.0f, 0.0f);
}

namespace {

// Ray (already in model space) against an axis-aligned box. Returns the entry distance or -1.
float slab(Vec3 o, Vec3 d, Vec3 boxLo, Vec3 boxHi) {
  float tMin = 0.0f, tMax = 1e9f;
  const float lo[3] = {boxLo.x, boxLo.y, boxLo.z};
  const float hi[3] = {boxHi.x, boxHi.y, boxHi.z};
  const float os[3] = {o.x, o.y, o.z};
  const float ds[3] = {d.x, d.y, d.z};
  for (int i = 0; i < 3; ++i) {
    if (std::fabs(ds[i]) < 1e-8f) {
      if (os[i] < lo[i] || os[i] > hi[i]) return -1.0f;
      continue;
    }
    float t0 = (lo[i] - os[i]) / ds[i], t1 = (hi[i] - os[i]) / ds[i];
    if (t0 > t1) std::swap(t0, t1);
    tMin = std::max(tMin, t0);
    tMax = std::min(tMax, t1);
    if (tMin > tMax) return -1.0f;
  }
  return tMin;
}

// Moves a world ray into the set's model space. Dividing by the scale keeps t in world units.
void toModel(const Pose& crtPose, float crtScale, Vec3 origin, Vec3 dir, Vec3* o, Vec3* d) {
  Pose inv = inverse(crtPose);
  *o = transformPoint(inv, origin) * (1.0f / crtScale);
  *d = rotate(inv.q, dir) * (1.0f / crtScale);
}

}  // namespace

float CrtScene::rayHit(const Pose& crtPose, float crtScale, Vec3 origin, Vec3 dir) const {
  Vec3 o, d;
  toModel(crtPose, crtScale, origin, dir, &o, &d);
  return slab(o, d, boundsMin_, boundsMax_);
}

int CrtScene::buttonAt(const Pose& crtPose, float crtScale, Vec3 origin, Vec3 dir) const {
  Vec3 o, d;
  toModel(crtPose, crtScale, origin, dir, &o, &d);
  // Only rays coming at the front panel can press anything.
  if (d.z >= -1e-6f) return -1;
  // Padding makes small real-world buttons easier to hit, more so vertically since they sit in a row.
  constexpr Vec3 pad{0.002f, 0.004f, 0.002f};
  int best = -1;
  float bestT = 1e9f;
  for (size_t i = 0; i < buttons_.size(); ++i) {
    const CrtButton& b = buttons_[i];
    float t = slab(o, d, Vec3{b.cx - b.hw, b.cy - b.hh, b.z0} - pad, Vec3{b.cx + b.hw, b.cy + b.hh, b.z1} + pad);
    if (t >= 0 && t < bestT) {
      bestT = t;
      best = static_cast<int>(i);
    }
  }
  return best;
}

int CrtScene::buttonUnderPoint(const Pose& crtPose, float crtScale, Vec3 point, float* depth) const {
  Vec3 p = transformPoint(inverse(crtPose), point) * (1.0f / crtScale);
  constexpr float pad = 0.003f;
  int best = -1;
  float bestDepth = -1e9f;
  for (size_t i = 0; i < buttons_.size(); ++i) {
    const CrtButton& b = buttons_[i];
    if (std::fabs(p.x - b.cx) > b.hw + pad || std::fabs(p.y - b.cy) > b.hh + pad) continue;
    // Only count a tip in front of the face or pushed a little way in, not one behind the panel.
    if (p.z < b.z0 - 0.02f || p.z > b.z1 + 0.04f) continue;
    float d = (b.z1 - p.z) * crtScale;
    if (d > bestDepth) {
      bestDepth = d;
      best = static_cast<int>(i);
    }
  }
  if (depth) *depth = bestDepth;
  return best;
}

bool CrtScene::nearFront(const Pose& crtPose, float crtScale, Vec3 point, float margin) const {
  Vec3 p = transformPoint(inverse(crtPose), point) * (1.0f / crtScale);
  float m = margin / crtScale;
  return p.x > boundsMin_.x - m && p.x < boundsMax_.x + m && p.y > boundsMin_.y - m &&
         p.y < boundsMax_.y + m && p.z > boundsMin_.z && p.z < boundsMax_.z + m;
}

void CrtScene::drawOccluders(const Mat4& viewProj, const float* spheres, int count) {
  if (count <= 0) return;
  count = std::min(count, kMaxOccluders);
  glUseProgram(occluderProgram_);
  glUniformMatrix4fv(glGetUniformLocation(occluderProgram_, "uViewProj"), 1, GL_FALSE, viewProj.m);
  glUniform4fv(glGetUniformLocation(occluderProgram_, "uSpheres"), count, spheres);
  glBindVertexArray(sphere_.vao);
  glDrawElementsInstanced(GL_TRIANGLES, sphere_.count, GL_UNSIGNED_INT, nullptr, count);
  glBindVertexArray(0);
}

void CrtScene::setHandMesh(int hand, const float* positions, const float* normals,
                           const int16_t* joints4, const float* weights4, int vertexCount,
                           const int16_t* indices, int indexCount) {
  freeMesh(handMeshes_[hand]);
  if (vertexCount <= 0 || indexCount <= 0) return;
  // pos3 nrm3 joints4 weights4
  std::vector<float> verts(static_cast<size_t>(vertexCount) * 14);
  for (int v = 0; v < vertexCount; ++v) {
    float* f = &verts[static_cast<size_t>(v) * 14];
    for (int k = 0; k < 3; ++k) {
      f[k] = positions[v * 3 + k];
      f[3 + k] = normals[v * 3 + k];
    }
    for (int k = 0; k < 4; ++k) {
      f[6 + k] = static_cast<float>(joints4[v * 4 + k]);
      f[10 + k] = weights4[v * 4 + k];
    }
  }
  std::vector<GLuint> tris(indexCount);
  for (int i = 0; i < indexCount; ++i) tris[i] = static_cast<uint16_t>(indices[i]);

  Mesh& m = handMeshes_[hand];
  glGenVertexArrays(1, &m.vao);
  glGenBuffers(1, &m.vbo);
  glGenBuffers(1, &m.ibo);
  glBindVertexArray(m.vao);
  glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
  glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, tris.size() * sizeof(GLuint), tris.data(), GL_STATIC_DRAW);
  const GLsizei stride = 14 * sizeof(float);
  const int sizes[] = {3, 3, 4, 4};
  size_t offset = 0;
  for (GLuint a = 0; a < 4; ++a) {
    glEnableVertexAttribArray(a);
    glVertexAttribPointer(a, sizes[a], GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<const void*>(offset * sizeof(float)));
    offset += sizes[a];
  }
  glBindVertexArray(0);
  m.count = static_cast<GLsizei>(tris.size());
}

void CrtScene::drawHandMesh(int hand, const Mat4& viewProj, const Mat4* skin, float inflate) {
  const Mesh& m = handMeshes_[hand];
  if (!m.count) return;
  glUseProgram(handProgram_);
  glUniformMatrix4fv(glGetUniformLocation(handProgram_, "uViewProj"), 1, GL_FALSE, viewProj.m);
  glUniformMatrix4fv(glGetUniformLocation(handProgram_, "uSkin"), kHandJoints, GL_FALSE, skin[0].m);
  glUniform1f(glGetUniformLocation(handProgram_, "uInflate"), inflate);
  glBindVertexArray(m.vao);
  glDrawElements(GL_TRIANGLES, m.count, GL_UNSIGNED_INT, nullptr);
  glBindVertexArray(0);
}
