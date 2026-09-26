// A TV loaded from a glTF 2.0 file, normalised into the CRT scene's model space: screen facing
// +Z, origin at the centre of the screen, metres.
#pragma once

#include <GLES3/gl3.h>

#include <cstdint>
#include <vector>

#include "crt_scene.h"
#include "xr_math.h"

class TvModel {
 public:
  // Loads a .glb/.gltf. The screen is the primitive whose material or node (or a parent node) is
  // named like "screen"; pressable buttons come from nodes named like "button".
  bool load(const char* path);
  void destroy();

  Vec3 boundsMin() const { return boundsMin_; }
  Vec3 boundsMax() const { return boundsMax_; }
  float screenHalfW() const { return screenHalfW_; }
  float screenHalfH() const { return screenHalfH_; }
  const std::vector<CrtButton>& buttons() const { return buttons_; }

  // How the video sits on the glass: it covers the glass's whole UV extent (overscan), with the
  // screen material's painted black surround and occlusion falloff composited over it.
  struct ScreenMask {
    GLuint texture = 0;          // painted surround (base colour); 0 for none
    float uvMin[2] = {0, 0};     // glass bounds in UV space
    float uvMax[2] = {1, 1};
    bool flipU = false, flipV = false;  // so video (0,0) is the bottom-left of the picture
    float threshold = 0;         // linear luminance separating mask from picture
    GLuint occlusion = 0;        // soft edge / corner falloff multiplied into the picture
    float occlusionStrength = 0;
  };
  const ScreenMask& screenMask() const { return mask_; }

  // Everything except the screen, with the PBR program bound by the caller.
  void drawBody(GLuint program, const Mat4& model, uint32_t hoveredButtons,
                uint32_t pressedButtons) const;
  // Just the screen glass, with the video program bound by the caller.
  void drawScreen() const;

 private:
  struct Material {
    GLuint base = 0, metallicRoughness = 0, normal = 0, occlusion = 0;
    float baseFactor[4] = {1, 1, 1, 1};
    float metallic = 1, roughness = 1, normalScale = 1, occlusionStrength = 0;
    bool hasNormalMap = false;
  };
  struct Draw {
    GLuint firstIndex = 0;
    GLsizei count = 0;
    int material = -1;
    int button = -1;  // index into buttons_, or -1
    bool screen = false;
    bool hasTangents = false;
  };

  GLuint vao_ = 0, vbo_ = 0, ibo_ = 0;
  GLuint white_ = 0, flatNormal_ = 0;
  std::vector<GLuint> textures_;
  std::vector<Material> materials_;
  std::vector<Draw> draws_;
  std::vector<CrtButton> buttons_;
  ScreenMask mask_;
  Vec3 boundsMin_, boundsMax_;
  float screenHalfW_ = 0, screenHalfH_ = 0;
};
