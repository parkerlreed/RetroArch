// Procedural TV/VCR combo with the mpv video mapped onto its curved tube.
#pragma once

#include <GLES3/gl3.h>

#include <cstdint>
#include <vector>

#include "xr_math.h"

struct Mesh {
  GLuint vao = 0, vbo = 0, ibo = 0;
  GLsizei count = 0;
};

struct ControllerVisual {
  bool active = false;
  bool drawController = true;  // false for tracked hands, which passthrough already shows
  Pose aim;
  float rayLength = 0;  // 0 hides the ray
  bool highlighted = false;
};

// A pressable button on the front panel, in model space.
struct CrtButton {
  int action;        // ActionCode fired when pressed
  float cx, cy;      // centre on the front face
  float hw, hh;      // half extents
  int icon;
  bool repeats;      // keeps firing while held
  float z0 = -0.004f, z1 = 0.008f;  // depth range, for hit testing
};

class TvModel;

class CrtScene {
 public:
  // Bounds of the procedural set. The screen faces +Z and the origin sits at the centre of the
  // tube face; a loaded model is normalised to the same convention.
  static constexpr Vec3 kBoundsMin{-0.25f, -0.30f, -0.44f};
  static constexpr Vec3 kBoundsMax{0.25f, 0.19f, 0.02f};
  static constexpr float kScreenHalfW = 0.20f;
  static constexpr float kScreenHalfH = 0.15f;
  static constexpr int kMaxButtons = 8;

  // Uses the glTF model at modelPath when it loads, otherwise the built-in TV/VCR.
  bool init(const char* modelPath);
  void destroy();

  void setVideo(GLuint externalTexture, const float* texMatrix, bool hasFrame);
  void setPlaying(bool playing) { playing_ = playing; }
  // Scanline strength 0..1, whether they fade out as they get too fine to show, and glass
  // reflection strength 0..1.
  void setPicture(float scanlines, bool scanlineFade, float reflections) {
    scanlines_ = scanlines;
    scanlineFade_ = scanlineFade;
    reflections_ = reflections;
  }
  // Bit i set means button i is hovered / held.
  void setButtonState(uint32_t hovered, uint32_t pressed) {
    hoveredButtons_ = hovered;
    pressedButtons_ = pressed;
  }

  void draw(const Mat4& viewProj, Vec3 eye, const Pose& crtPose, float crtScale,
            const ControllerVisual* controllers, int controllerCount);

  // Ray against the set's bounding box. Returns the hit distance along dir, or a negative value.
  float rayHit(const Pose& crtPose, float crtScale, Vec3 origin, Vec3 dir) const;
  // Index of the front-panel button the ray points at, or -1.
  int buttonAt(const Pose& crtPose, float crtScale, Vec3 origin, Vec3 dir) const;
  const CrtButton& button(int index) const { return buttons_[index]; }
  // Button whose face a fingertip is over, or -1. *depth is how far (world metres) the tip is
  // pushed past the face: negative while still in front of it.
  int buttonUnderPoint(const Pose& crtPose, float crtScale, Vec3 point, float* depth) const;
  // True when a point is within `margin` metres in front of, or inside, the set.
  bool nearFront(const Pose& crtPose, float crtScale, Vec3 point, float margin) const;

  static constexpr int kMaxOccluders = 128;
  // Punches passthrough-coloured holes where tracked hands are in front of the set, so the real
  // hands stay visible. `spheres` holds xyz + radius per sphere, in world space.
  void drawOccluders(const Mat4& viewProj, const float* spheres, int count);

  // The runtime's skinned hand model, used as a closer-fitting occluder than spheres.
  // `joints4`/`weights4` are four blend joints and weights per vertex.
  static constexpr int kHandJoints = 26;
  void setHandMesh(int hand, const float* positions, const float* normals, const int16_t* joints4,
                   const float* weights4, int vertexCount, const int16_t* indices, int indexCount);
  bool hasHandMesh(int hand) const { return handMeshes_[hand].count > 0; }
  // `skin` holds one matrix per joint: current joint pose * inverse bind pose.
  void drawHandMesh(int hand, const Mat4& viewProj, const Mat4* skin, float inflate);
  // Screen diagonal at scale 1, for the size presets.
  float screenDiagonalInches() const;
  float screenAspect() const { return screenHalfW_ / screenHalfH_; }

 private:
  GLuint litProgram_ = 0, screenProgram_ = 0, pbrProgram_ = 0, occluderProgram_ = 0;
  GLuint handProgram_ = 0;
  Mesh sphere_;
  Mesh handMeshes_[2];
  TvModel* model_ = nullptr;
  Vec3 boundsMin_ = kBoundsMin, boundsMax_ = kBoundsMax;
  float screenHalfW_ = kScreenHalfW, screenHalfH_ = kScreenHalfH;
  std::vector<CrtButton> buttons_;
  Mesh body_, trim_, slot_, accent_, led_, screen_, box_;
  Mesh buttonBodies_[kMaxButtons], buttonIcons_[kMaxButtons];
  GLuint videoTexture_ = 0;
  float texMatrix_[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  bool hasFrame_ = false;
  bool playing_ = false;
  float scanlines_ = 0.35f;
  bool scanlineFade_ = true;
  float reflections_ = 1.0f;
  uint32_t hoveredButtons_ = 0, pressedButtons_ = 0;

  void drawProcedural(const Mat4& model, const Mat4& viewProj, Vec3 eye);
  void drawLit(const Mesh& mesh, const Mat4& model, const Mat4& viewProj, Vec3 eye, Vec3 color,
               float emissive, float spec);
};
