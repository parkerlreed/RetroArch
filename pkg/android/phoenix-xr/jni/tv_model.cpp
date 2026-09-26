#include "tv_model.h"

#include <GLES2/gl2ext.h>
#include <android/log.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>

#include "xr_actions.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#define CGLTF_IMPLEMENTATION
#include "third_party/cgltf.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#include "third_party/stb_image.h"
#pragma clang diagnostic pop

#define LOG_TAG "RetroArch-XR"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

constexpr int kFloatsPerVertex = 12;  // pos3 nrm3 uv2 tan4
constexpr float kButtonTravel = 0.0018f;

bool nameContains(const char* name, const char* needle) {
  if (!name) return false;
  std::string n(name);
  std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return std::tolower(c); });
  return n.find(needle) != std::string::npos;
}

// True if the node or any of its parents is named like `needle`.
bool nodeTagged(const cgltf_node* node, const char* needle) {
  for (; node; node = node->parent)
    if (nameContains(node->name, needle)) return true;
  return false;
}

GLuint solidTexture(uint8_t r, uint8_t g, uint8_t b) {
  const uint8_t px[4] = {r, g, b, 255};
  GLuint t;
  glGenTextures(1, &t);
  glBindTexture(GL_TEXTURE_2D, t);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  return t;
}

GLuint uploadImage(const cgltf_image* image, bool srgb, float anisotropy) {
  if (!image->buffer_view) {
    LOGE("image %s is not embedded; only .glb / embedded images are supported",
         image->uri ? image->uri : "?");
    return 0;
  }
  const auto* bytes = static_cast<const stbi_uc*>(cgltf_buffer_view_data(image->buffer_view));
  int w = 0, h = 0, comp = 0;
  stbi_uc* pixels = stbi_load_from_memory(bytes, static_cast<int>(image->buffer_view->size), &w, &h,
                                          &comp, 4);
  if (!pixels) {
    LOGE("failed to decode image: %s", stbi_failure_reason());
    return 0;
  }
  GLuint t;
  glGenTextures(1, &t);
  glBindTexture(GL_TEXTURE_2D, t);
  // glTF UVs have their origin at the top-left, which matches uploading row 0 first.
  glTexImage2D(GL_TEXTURE_2D, 0, srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8, w, h, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, pixels);
  stbi_image_free(pixels);
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  if (anisotropy > 1.0f) glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy);
  return t;
}

// Inverse-transpose of the upper 3x3 of a column-major 4x4, for transforming normals.
void normalMatrix(const float* m, float out[9], float* det) {
  float a = m[0], b = m[4], c = m[8];
  float d = m[1], e = m[5], f = m[9];
  float g = m[2], h = m[6], i = m[10];
  float A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  float D = -(b * i - c * h), E = a * i - c * g, F = -(a * h - b * g);
  float G = b * f - c * e, H = -(a * f - c * d), I = a * e - b * d;
  *det = a * A + b * B + c * C;
  float inv = std::fabs(*det) > 1e-20f ? 1.0f / *det : 0.0f;
  // Cofactor matrix / det is the inverse-transpose; store column-major.
  out[0] = A * inv; out[1] = B * inv; out[2] = C * inv;
  out[3] = D * inv; out[4] = E * inv; out[5] = F * inv;
  out[6] = G * inv; out[7] = H * inv; out[8] = I * inv;
}

Vec3 mul3(const float* m9, Vec3 v) {  // column-major 3x3
  return {m9[0] * v.x + m9[3] * v.y + m9[6] * v.z, m9[1] * v.x + m9[4] * v.y + m9[7] * v.z,
          m9[2] * v.x + m9[5] * v.y + m9[8] * v.z};
}

Vec3 mulPoint(const float* m, Vec3 v) {
  return {m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12], m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13],
          m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14]};
}

Vec3 mulDir(const float* m, Vec3 v) {
  return {m[0] * v.x + m[4] * v.y + m[8] * v.z, m[1] * v.x + m[5] * v.y + m[9] * v.z,
          m[2] * v.x + m[6] * v.y + m[10] * v.z};
}

struct PrimitiveData {
  std::vector<float> verts;       // kFloatsPerVertex each, in glTF world space
  std::vector<uint32_t> indices;  // local to verts
  const cgltf_material* material = nullptr;
  bool screen = false;
  bool buttons = false;
  bool hasTangents = false;
};

// Splits a triangle list into connected pieces (vertices welded by position).
std::vector<std::vector<uint32_t>> connectedPieces(const PrimitiveData& p) {
  size_t vcount = p.verts.size() / kFloatsPerVertex;
  std::vector<uint32_t> parent(vcount);
  for (size_t i = 0; i < vcount; ++i) parent[i] = static_cast<uint32_t>(i);
  auto find = [&](uint32_t x) {
    while (parent[x] != x) x = parent[x] = parent[parent[x]];
    return x;
  };
  std::map<std::tuple<long, long, long>, uint32_t> welded;
  for (size_t i = 0; i < vcount; ++i) {
    const float* v = &p.verts[i * kFloatsPerVertex];
    auto key = std::make_tuple(std::lround(v[0] * 1e5), std::lround(v[1] * 1e5), std::lround(v[2] * 1e5));
    auto it = welded.emplace(key, static_cast<uint32_t>(i)).first;
    parent[find(static_cast<uint32_t>(i))] = find(it->second);
  }
  for (size_t t = 0; t + 2 < p.indices.size(); t += 3) {
    uint32_t a = find(p.indices[t]), b = find(p.indices[t + 1]), c = find(p.indices[t + 2]);
    parent[a] = b;
    parent[find(c)] = find(b);
  }
  std::unordered_map<uint32_t, std::vector<uint32_t>> groups;
  for (size_t t = 0; t + 2 < p.indices.size(); t += 3) {
    auto& g = groups[find(p.indices[t])];
    g.insert(g.end(), {p.indices[t], p.indices[t + 1], p.indices[t + 2]});
  }
  std::vector<std::vector<uint32_t>> out;
  for (auto& [root, tris] : groups) out.push_back(std::move(tris));
  return out;
}

}  // namespace

bool TvModel::load(const char* path) {
  cgltf_options options{};
  cgltf_data* data = nullptr;
  if (cgltf_parse_file(&options, path, &data) != cgltf_result_success) {
    LOGI("no TV model at %s", path);
    return false;
  }
  if (cgltf_load_buffers(&options, data, path) != cgltf_result_success) {
    LOGE("failed to load buffers for %s", path);
    cgltf_free(data);
    return false;
  }

  // 1. Gather every triangle primitive in glTF world space.
  std::vector<PrimitiveData> prims;
  for (size_t n = 0; n < data->nodes_count; ++n) {
    const cgltf_node* node = &data->nodes[n];
    if (!node->mesh) continue;
    float world[16];
    cgltf_node_transform_world(node, world);
    float nrm[9], det;
    normalMatrix(world, nrm, &det);
    float handedness = det < 0 ? -1.0f : 1.0f;

    for (size_t pi = 0; pi < node->mesh->primitives_count; ++pi) {
      const cgltf_primitive& prim = node->mesh->primitives[pi];
      if (prim.type != cgltf_primitive_type_triangles) continue;
      const cgltf_accessor *pos = nullptr, *normal = nullptr, *uv = nullptr, *tangent = nullptr;
      for (size_t a = 0; a < prim.attributes_count; ++a) {
        const cgltf_attribute& attr = prim.attributes[a];
        if (attr.type == cgltf_attribute_type_position) pos = attr.data;
        else if (attr.type == cgltf_attribute_type_normal) normal = attr.data;
        else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) uv = attr.data;
        else if (attr.type == cgltf_attribute_type_tangent) tangent = attr.data;
      }
      if (!pos) continue;

      PrimitiveData p;
      p.material = prim.material;
      p.screen = nodeTagged(node, "screen") || (prim.material && nameContains(prim.material->name, "screen"));
      p.buttons = !p.screen && nodeTagged(node, "button");
      p.hasTangents = tangent != nullptr;
      p.verts.resize(pos->count * kFloatsPerVertex);
      for (size_t v = 0; v < pos->count; ++v) {
        float* out = &p.verts[v * kFloatsPerVertex];
        float tmp[4] = {0, 0, 0, 1};
        cgltf_accessor_read_float(pos, v, tmp, 3);
        Vec3 wp = mulPoint(world, {tmp[0], tmp[1], tmp[2]});
        Vec3 wn{0, 1, 0};
        if (normal) {
          cgltf_accessor_read_float(normal, v, tmp, 3);
          wn = normalize(mul3(nrm, {tmp[0], tmp[1], tmp[2]}));
        }
        float u = 0, w = 0;
        if (uv) {
          cgltf_accessor_read_float(uv, v, tmp, 2);
          u = tmp[0];
          w = tmp[1];
        }
        Vec3 wt{1, 0, 0};
        float tw = 1;
        if (tangent) {
          cgltf_accessor_read_float(tangent, v, tmp, 4);
          wt = normalize(mulDir(world, {tmp[0], tmp[1], tmp[2]}));
          tw = tmp[3] * handedness;
        }
        const float vals[kFloatsPerVertex] = {wp.x, wp.y, wp.z, wn.x, wn.y, wn.z, u, w, wt.x, wt.y, wt.z, tw};
        std::copy(vals, vals + kFloatsPerVertex, out);
      }
      if (prim.indices) {
        p.indices.resize(prim.indices->count);
        for (size_t i = 0; i < prim.indices->count; ++i)
          p.indices[i] = static_cast<uint32_t>(cgltf_accessor_read_index(prim.indices, i));
      } else {
        p.indices.resize(pos->count);
        for (size_t i = 0; i < pos->count; ++i) p.indices[i] = static_cast<uint32_t>(i);
      }
      prims.push_back(std::move(p));
    }
  }

  // 2. Find the screen and work out how to face it down +Z with its centre at the origin.
  Vec3 screenNormal{0, 0, 0};
  bool haveScreen = false;
  for (const PrimitiveData& p : prims) {
    if (!p.screen) continue;
    haveScreen = true;
    for (size_t v = 0; v < p.verts.size(); v += kFloatsPerVertex)
      screenNormal = screenNormal + Vec3{p.verts[v + 3], p.verts[v + 4], p.verts[v + 5]};
  }
  if (!haveScreen) {
    LOGE("%s has no primitive named like 'screen'", path);
    cgltf_free(data);
    return false;
  }
  screenNormal = normalize(screenNormal);
  Quat facing = axisAngle({0, 1, 0}, -std::atan2(screenNormal.x, screenNormal.z));

  auto apply = [&](PrimitiveData& p) {
    for (size_t v = 0; v < p.verts.size(); v += kFloatsPerVertex) {
      float* f = &p.verts[v];
      Vec3 a = rotate(facing, {f[0], f[1], f[2]});
      Vec3 n = rotate(facing, {f[3], f[4], f[5]});
      Vec3 t = rotate(facing, {f[8], f[9], f[10]});
      f[0] = a.x; f[1] = a.y; f[2] = a.z;
      f[3] = n.x; f[4] = n.y; f[5] = n.z;
      f[8] = t.x; f[9] = t.y; f[10] = t.z;
    }
  };
  for (PrimitiveData& p : prims) apply(p);

  Vec3 sMin{1e9f, 1e9f, 1e9f}, sMax{-1e9f, -1e9f, -1e9f};
  for (const PrimitiveData& p : prims) {
    if (!p.screen) continue;
    for (size_t v = 0; v < p.verts.size(); v += kFloatsPerVertex) {
      sMin = {std::min(sMin.x, p.verts[v]), std::min(sMin.y, p.verts[v + 1]), std::min(sMin.z, p.verts[v + 2])};
      sMax = {std::max(sMax.x, p.verts[v]), std::max(sMax.y, p.verts[v + 1]), std::max(sMax.z, p.verts[v + 2])};
    }
  }
  Vec3 centre = (sMin + sMax) * 0.5f;
  screenHalfW_ = (sMax.x - sMin.x) * 0.5f;
  screenHalfH_ = (sMax.y - sMin.y) * 0.5f;

  boundsMin_ = {1e9f, 1e9f, 1e9f};
  boundsMax_ = {-1e9f, -1e9f, -1e9f};
  for (PrimitiveData& p : prims) {
    for (size_t v = 0; v < p.verts.size(); v += kFloatsPerVertex) {
      float* f = &p.verts[v];
      f[0] -= centre.x;
      f[1] -= centre.y;
      f[2] -= centre.z;
      boundsMin_ = {std::min(boundsMin_.x, f[0]), std::min(boundsMin_.y, f[1]), std::min(boundsMin_.z, f[2])};
      boundsMax_ = {std::max(boundsMax_.x, f[0]), std::max(boundsMax_.y, f[1]), std::max(boundsMax_.z, f[2])};
    }
  }

  // 3. Textures and materials.
  float anisotropy = 1.0f;
  const char* ext = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
  if (ext && strstr(ext, "GL_EXT_texture_filter_anisotropic")) {
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &anisotropy);
    anisotropy = std::min(anisotropy, 8.0f);
  }
  white_ = solidTexture(255, 255, 255);
  flatNormal_ = solidTexture(128, 128, 255);
  std::map<std::pair<const cgltf_image*, bool>, GLuint> imageCache;
  auto texture = [&](const cgltf_texture* tex, bool srgb, GLuint fallback) {
    if (!tex || !tex->image) return fallback;
    auto key = std::make_pair(static_cast<const cgltf_image*>(tex->image), srgb);
    auto it = imageCache.find(key);
    if (it != imageCache.end()) return it->second;
    GLuint t = uploadImage(tex->image, srgb, anisotropy);
    if (t) textures_.push_back(t);
    imageCache[key] = t ? t : fallback;
    return t ? t : fallback;
  };
  std::map<const cgltf_material*, int> materialIndex;
  auto material = [&](const cgltf_material* m) {
    auto it = materialIndex.find(m);
    if (it != materialIndex.end()) return it->second;
    Material mat;
    mat.base = mat.metallicRoughness = mat.occlusion = white_;
    mat.normal = flatNormal_;
    if (m) {
      if (m->has_pbr_metallic_roughness) {
        const auto& pbr = m->pbr_metallic_roughness;
        std::copy(pbr.base_color_factor, pbr.base_color_factor + 4, mat.baseFactor);
        mat.metallic = pbr.metallic_factor;
        mat.roughness = pbr.roughness_factor;
        mat.base = texture(pbr.base_color_texture.texture, true, white_);
        mat.metallicRoughness = texture(pbr.metallic_roughness_texture.texture, false, white_);
      }
      if (m->normal_texture.texture) {
        mat.normal = texture(m->normal_texture.texture, false, flatNormal_);
        mat.normalScale = m->normal_texture.scale;
        mat.hasNormalMap = true;
      }
      if (m->occlusion_texture.texture) {
        mat.occlusion = texture(m->occlusion_texture.texture, false, white_);
        mat.occlusionStrength = m->occlusion_texture.scale;
      }
    }
    materials_.push_back(mat);
    int index = static_cast<int>(materials_.size() - 1);
    materialIndex[m] = index;
    return index;
  };

  // 3b. Screen compositing. The picture is fitted to the lit area inside the black surround
  // painted into the screen's base colour (cores already crop their own overscan); the
  // surround and the occlusion falloff are then composited over it.
  for (const PrimitiveData& p : prims) {
    if (!p.screen) continue;

    // UV extent of the glass, and which way U and V run across it.
    float uMin = 1e9f, uMax = -1e9f, vMin = 1e9f, vMax = -1e9f;
    double su = 0, sv = 0, sx = 0, sy = 0, n = 0;
    for (size_t v = 0; v < p.verts.size(); v += kFloatsPerVertex) {
      const float* f = &p.verts[v];
      uMin = std::min(uMin, f[6]); uMax = std::max(uMax, f[6]);
      vMin = std::min(vMin, f[7]); vMax = std::max(vMax, f[7]);
      su += f[6]; sv += f[7]; sx += f[0]; sy += f[1]; n += 1;
    }
    double cux = 0, cvy = 0;
    for (size_t v = 0; v < p.verts.size(); v += kFloatsPerVertex) {
      const float* f = &p.verts[v];
      cux += (f[6] - su / n) * (f[0] - sx / n);
      cvy += (f[7] - sv / n) * (f[1] - sy / n);
    }
    mask_.uvMin[0] = uMin;
    mask_.uvMin[1] = vMin;
    mask_.uvMax[0] = uMax;
    mask_.uvMax[1] = vMax;
    mask_.flipU = cux < 0;
    mask_.flipV = cvy < 0;

    const cgltf_material* m = p.material;
    if (m && m->occlusion_texture.texture) {
      mask_.occlusion = texture(m->occlusion_texture.texture, false, white_);
      mask_.occlusionStrength = m->occlusion_texture.scale;
    }

    // The painted surround is much darker than the picture area, so threshold halfway between.
    const cgltf_texture* tex =
        m && m->has_pbr_metallic_roughness ? m->pbr_metallic_roughness.base_color_texture.texture : nullptr;
    if (!tex || !tex->image || !tex->image->buffer_view) break;
    const auto* bytes = static_cast<const stbi_uc*>(cgltf_buffer_view_data(tex->image->buffer_view));
    int w = 0, h = 0, comp = 0;
    stbi_uc* px = stbi_load_from_memory(bytes, static_cast<int>(tex->image->buffer_view->size), &w,
                                        &h, &comp, 4);
    if (!px) break;
    auto clampi = [](int x, int lo, int hi) { return std::max(lo, std::min(hi, x)); };
    int x0 = clampi(static_cast<int>(std::floor(uMin * w)), 0, w - 1);
    int x1 = clampi(static_cast<int>(std::ceil(uMax * w)) - 1, 0, w - 1);
    int y0 = clampi(static_cast<int>(std::floor(vMin * h)), 0, h - 1);
    int y1 = clampi(static_cast<int>(std::ceil(vMax * h)) - 1, 0, h - 1);
    int lo = 255, hi = 0;
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) {
        const stbi_uc* q = &px[(y * w + x) * 4];
        int l = (q[0] + q[1] + q[2]) / 3;
        lo = std::min(lo, l);
        hi = std::max(hi, l);
      }
    if (hi - lo >= 12) {
      int t = (lo + hi) / 2;
      int bx0 = x1, bx1 = x0, by0 = y1, by1 = y0;
      for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) {
          const stbi_uc* q = &px[(y * w + x) * 4];
          if ((q[0] + q[1] + q[2]) / 3 > t) {
            bx0 = std::min(bx0, x); bx1 = std::max(bx1, x);
            by0 = std::min(by0, y); by1 = std::max(by1, y);
          }
        }
      // The mask is sampled from the sRGB base texture, so compare in linear.
      mask_.threshold = std::pow(t / 255.0f, 2.2f);
      mask_.texture = texture(tex, true, white_);
      if (bx1 > bx0 && by1 > by0) {
        mask_.uvMin[0] = static_cast<float>(bx0) / w;
        mask_.uvMin[1] = static_cast<float>(by0) / h;
        mask_.uvMax[0] = static_cast<float>(bx1 + 1) / w;
        mask_.uvMax[1] = static_cast<float>(by1 + 1) / h;
        // The picture, not the whole glass, sets the video aspect and the TV's "size".
        float metresPerU = 2 * screenHalfW_ / (uMax - uMin);
        float metresPerV = 2 * screenHalfH_ / (vMax - vMin);
        screenHalfW_ = (mask_.uvMax[0] - mask_.uvMin[0]) * metresPerU * 0.5f;
        screenHalfH_ = (mask_.uvMax[1] - mask_.uvMin[1]) * metresPerV * 0.5f;
      }
    }
    stbi_image_free(px);
    LOGI("screen: picture uv (%.3f,%.3f)-(%.3f,%.3f), flip %d/%d, mask %s, occlusion %s",
         mask_.uvMin[0], mask_.uvMin[1], mask_.uvMax[0], mask_.uvMax[1], mask_.flipU, mask_.flipV,
         mask_.texture ? "yes" : "no", mask_.occlusion ? "yes" : "no");
    break;
  }

  // 4. One vertex/index buffer; draws are index ranges.
  std::vector<float> allVerts;
  std::vector<GLuint> allIndices;
  struct Piece {
    std::vector<uint32_t> tris;
    Vec3 lo, hi;
    int material;
    bool hasTangents;
  };
  std::vector<Piece> buttonPieces;

  for (PrimitiveData& p : prims) {
    GLuint base = static_cast<GLuint>(allVerts.size() / kFloatsPerVertex);
    allVerts.insert(allVerts.end(), p.verts.begin(), p.verts.end());
    int mat = material(p.material);
    if (p.buttons) {
      for (auto& tris : connectedPieces(p)) {
        Piece piece{{}, {1e9f, 1e9f, 1e9f}, {-1e9f, -1e9f, -1e9f}, mat, p.hasTangents};
        for (uint32_t i : tris) {
          const float* f = &p.verts[i * kFloatsPerVertex];
          piece.lo = {std::min(piece.lo.x, f[0]), std::min(piece.lo.y, f[1]), std::min(piece.lo.z, f[2])};
          piece.hi = {std::max(piece.hi.x, f[0]), std::max(piece.hi.y, f[1]), std::max(piece.hi.z, f[2])};
          piece.tris.push_back(i + base);
        }
        buttonPieces.push_back(std::move(piece));
      }
      continue;
    }
    Draw d;
    d.firstIndex = static_cast<GLuint>(allIndices.size());
    for (uint32_t i : p.indices) allIndices.push_back(i + base);
    d.count = static_cast<GLsizei>(p.indices.size());
    d.material = mat;
    d.screen = p.screen;
    d.hasTangents = p.hasTangents;
    draws_.push_back(d);
  }

  // Buttons, left to right as seen from the front.
  std::sort(buttonPieces.begin(), buttonPieces.end(),
            [](const Piece& a, const Piece& b) { return a.lo.x + a.hi.x < b.lo.x + b.hi.x; });
  // The layout printed on the Sketchfab "CRT TV" front panel:
  // POWER, VOLUME < >, CHANNEL ^ v, ACTION, TV/VIDEO.
  static const struct { int action; bool repeats; } kSevenButtonLayout[] = {
      {kExit, false},         {kVolumeDown, true},   {kVolumeUp, true},       {kSeekForwardLong, true},
      {kSeekBackLong, true},  {kTogglePause, false}, {kShowProgress, false},
  };
  bool mapped = buttonPieces.size() == 7;
  if (!buttonPieces.empty() && !mapped)
    LOGI("model has %zu buttons; only the 7-button layout is mapped, leaving them inert",
         buttonPieces.size());
  for (size_t i = 0; i < buttonPieces.size(); ++i) {
    Piece& piece = buttonPieces[i];
    Draw d;
    d.firstIndex = static_cast<GLuint>(allIndices.size());
    allIndices.insert(allIndices.end(), piece.tris.begin(), piece.tris.end());
    d.count = static_cast<GLsizei>(piece.tris.size());
    d.material = piece.material;
    d.hasTangents = piece.hasTangents;
    if (mapped && buttons_.size() < CrtScene::kMaxButtons) {
      CrtButton b{};
      b.action = kSevenButtonLayout[i].action;
      b.repeats = kSevenButtonLayout[i].repeats;
      b.cx = (piece.lo.x + piece.hi.x) * 0.5f;
      b.cy = (piece.lo.y + piece.hi.y) * 0.5f;
      b.hw = (piece.hi.x - piece.lo.x) * 0.5f;
      b.hh = (piece.hi.y - piece.lo.y) * 0.5f;
      b.z0 = piece.lo.z;
      b.z1 = piece.hi.z;
      d.button = static_cast<int>(buttons_.size());
      buttons_.push_back(b);
    }
    draws_.push_back(d);
  }

  glGenVertexArrays(1, &vao_);
  glGenBuffers(1, &vbo_);
  glGenBuffers(1, &ibo_);
  glBindVertexArray(vao_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, allVerts.size() * sizeof(float), allVerts.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, allIndices.size() * sizeof(GLuint), allIndices.data(),
               GL_STATIC_DRAW);
  const GLsizei stride = kFloatsPerVertex * sizeof(float);
  const int sizes[] = {3, 3, 2, 4};
  size_t offset = 0;
  for (GLuint a = 0; a < 4; ++a) {
    glEnableVertexAttribArray(a);
    glVertexAttribPointer(a, sizes[a], GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<const void*>(offset * sizeof(float)));
    offset += sizes[a];
  }
  glBindVertexArray(0);

  LOGI("loaded %s: %zu vertices, %zu triangles, %zu materials, %zu buttons, screen %.3fx%.3f m",
       path, allVerts.size() / kFloatsPerVertex, allIndices.size() / 3, materials_.size(),
       buttons_.size(), screenHalfW_ * 2, screenHalfH_ * 2);
  cgltf_free(data);
  return true;
}

void TvModel::destroy() {
  if (vao_) glDeleteVertexArrays(1, &vao_);
  if (vbo_) glDeleteBuffers(1, &vbo_);
  if (ibo_) glDeleteBuffers(1, &ibo_);
  if (!textures_.empty()) glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
  if (white_) glDeleteTextures(1, &white_);
  if (flatNormal_) glDeleteTextures(1, &flatNormal_);
  vao_ = vbo_ = ibo_ = white_ = flatNormal_ = 0;
  textures_.clear();
  materials_.clear();
  draws_.clear();
  buttons_.clear();
}

void TvModel::drawBody(GLuint program, const Mat4& model, uint32_t hoveredButtons,
                       uint32_t pressedButtons) const {
  const GLint uModel = glGetUniformLocation(program, "uModel");
  const GLint uBaseFactor = glGetUniformLocation(program, "uBaseFactor");
  const GLint uMetallic = glGetUniformLocation(program, "uMetallic");
  const GLint uRoughness = glGetUniformLocation(program, "uRoughness");
  const GLint uNormalScale = glGetUniformLocation(program, "uNormalScale");
  const GLint uUseNormalMap = glGetUniformLocation(program, "uUseNormalMap");
  const GLint uOcclusion = glGetUniformLocation(program, "uOcclusionStrength");
  const GLint uHighlight = glGetUniformLocation(program, "uHighlight");
  glUniform1i(glGetUniformLocation(program, "uBase"), 0);
  glUniform1i(glGetUniformLocation(program, "uMetalRough"), 1);
  glUniform1i(glGetUniformLocation(program, "uNormalMap"), 2);
  glUniform1i(glGetUniformLocation(program, "uOcclusionMap"), 3);

  glBindVertexArray(vao_);
  for (const Draw& d : draws_) {
    if (d.screen) continue;
    const Material& m = materials_[d.material];
    bool hovered = d.button >= 0 && (hoveredButtons & (1u << d.button));
    bool pressed = d.button >= 0 && (pressedButtons & (1u << d.button));
    Mat4 at = model;
    if (pressed) {
      // Push the button back into the panel along the model's -Z.
      Mat4 push = Mat4::identity();
      push.m[14] = -kButtonTravel;
      at = model * push;
    }
    glUniformMatrix4fv(uModel, 1, GL_FALSE, at.m);
    glUniform4fv(uBaseFactor, 1, m.baseFactor);
    glUniform1f(uMetallic, m.metallic);
    glUniform1f(uRoughness, m.roughness);
    glUniform1f(uNormalScale, m.normalScale);
    glUniform1f(uUseNormalMap, m.hasNormalMap && d.hasTangents ? 1.0f : 0.0f);
    glUniform1f(uOcclusion, m.occlusionStrength);
    glUniform1f(uHighlight, hovered ? 1.0f : 0.0f);
    const GLuint units[] = {m.base, m.metallicRoughness, m.normal, m.occlusion};
    for (int u = 0; u < 4; ++u) {
      glActiveTexture(GL_TEXTURE0 + u);
      glBindTexture(GL_TEXTURE_2D, units[u]);
    }
    glDrawElements(GL_TRIANGLES, d.count, GL_UNSIGNED_INT,
                   reinterpret_cast<const void*>(d.firstIndex * sizeof(GLuint)));
  }
  glActiveTexture(GL_TEXTURE0);
  glBindVertexArray(0);
}

void TvModel::drawScreen() const {
  glBindVertexArray(vao_);
  for (const Draw& d : draws_) {
    if (!d.screen) continue;
    glDrawElements(GL_TRIANGLES, d.count, GL_UNSIGNED_INT,
                   reinterpret_cast<const void*>(d.firstIndex * sizeof(GLuint)));
  }
  glBindVertexArray(0);
}
