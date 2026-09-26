// OpenXR passthrough session that shows mpv's output on a CRT placed in the room.
//
// Kotlin owns mpv and the SurfaceTexture; this file owns EGL, the OpenXR session, input and
// rendering. Everything here runs on the single render thread that calls XrNative.run().

#include <jni.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <android/log.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "crt_scene.h"
#include "xr_actions.h"
#include "xr_math.h"

#define LOG_TAG "RetroArch-XR"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define XR_CHECK(expr)                                        \
  do {                                                        \
    XrResult r_ = (expr);                                     \
    if (XR_FAILED(r_)) {                                      \
      LOGE("%s failed: %d (%s:%d)", #expr, r_, __FILE__, __LINE__); \
      return false;                                           \
    }                                                         \
  } while (0)

namespace {

std::atomic<bool> gExitRequested{false};

constexpr float kSizePresetsInches[] = {14, 20, 27, 32, 40};

constexpr int kLeft = 0, kRight = 1;

// VR preferences from Kotlin, in XrPlayerActivity.nativeSettings() order.
struct XrSettings {
  float handPadding = 0.004f;        // metres the hand cutout is grown by
  float passthroughOpacity = 1.0f;   // 1 - room dimming
  float passthroughSaturation = 1.0f;
  float scanlines = 0.35f;
  bool scanlineFade = true;
  float reflections = 1.0f;
  float reachDistance = 0.6f;        // where the menu gesture puts the set

  void read(JNIEnv* env, jfloatArray array) {
    if (!array) return;
    float v[7];
    jsize n = std::min<jsize>(env->GetArrayLength(array), 7);
    env->GetFloatArrayRegion(array, 0, n, v);
    if (n > 0) handPadding = std::clamp(v[0], 0.0f, 0.03f);
    if (n > 1) passthroughOpacity = std::clamp(v[1], 0.0f, 1.0f);
    if (n > 2) passthroughSaturation = std::clamp(v[2], 0.0f, 1.0f);
    if (n > 3) scanlines = std::clamp(v[3], 0.0f, 1.0f);
    if (n > 4) scanlineFade = v[4] > 0.5f;
    if (n > 5) reflections = std::clamp(v[5], 0.0f, 1.0f);
    if (n > 6) reachDistance = std::clamp(v[6], 0.3f, 1.5f);
  }
};

struct Bridge {
  JNIEnv* env = nullptr;
  jobject obj = nullptr;
  jmethodID onGlReady = nullptr;
  jmethodID updateVideoTexture = nullptr;
  jmethodID isPlaying = nullptr;
  jmethodID onAction = nullptr;
  jmethodID onCrtPoseChanged = nullptr;
  jmethodID onSessionEnded = nullptr;
  jfloatArray matrix = nullptr;

  bool init(JNIEnv* e, jobject bridge) {
    env = e;
    obj = bridge;
    jclass cls = env->GetObjectClass(bridge);
    onGlReady = env->GetMethodID(cls, "onGlReady", "(IF)V");
    updateVideoTexture = env->GetMethodID(cls, "updateVideoTexture", "([F)Z");
    isPlaying = env->GetMethodID(cls, "isPlaying", "()Z");
    onAction = env->GetMethodID(cls, "onAction", "(I)V");
    onCrtPoseChanged = env->GetMethodID(cls, "onCrtPoseChanged", "([F)V");
    onSessionEnded = env->GetMethodID(cls, "onSessionEnded", "()V");
    env->DeleteLocalRef(cls);
    if (env->ExceptionCheck()) {
      env->ExceptionDescribe();
      env->ExceptionClear();
      return false;
    }
    matrix = static_cast<jfloatArray>(env->NewGlobalRef(env->NewFloatArray(16)));
    return true;
  }

  void release() {
    if (matrix) env->DeleteGlobalRef(matrix);
    matrix = nullptr;
  }

  void check() {
    if (env->ExceptionCheck()) {
      env->ExceptionDescribe();
      env->ExceptionClear();
    }
  }

  void glReady(GLuint texture, float screenAspect) {
    env->CallVoidMethod(obj, onGlReady, static_cast<jint>(texture), static_cast<jfloat>(screenAspect));
    check();
  }

  bool updateVideo(float* out) {
    bool hasFrame = env->CallBooleanMethod(obj, updateVideoTexture, matrix);
    check();
    env->GetFloatArrayRegion(matrix, 0, 16, out);
    return hasFrame;
  }

  bool playing() {
    bool p = env->CallBooleanMethod(obj, isPlaying);
    check();
    return p;
  }

  void action(jint code) {
    env->CallVoidMethod(obj, onAction, code);
    check();
  }

  void crtPoseChanged(const Pose& pose, float scale) {
    float v[8] = {pose.p.x, pose.p.y, pose.p.z, pose.q.x, pose.q.y, pose.q.z, pose.q.w, scale};
    jfloatArray arr = env->NewFloatArray(8);
    env->SetFloatArrayRegion(arr, 0, 8, v);
    env->CallVoidMethod(obj, onCrtPoseChanged, arr);
    env->DeleteLocalRef(arr);
    check();
  }

  void sessionEnded() {
    env->CallVoidMethod(obj, onSessionEnded);
    check();
  }
};

struct EyeTarget {
  XrSwapchain swapchain = XR_NULL_HANDLE;
  int32_t width = 0, height = 0;
  std::vector<XrSwapchainImageOpenGLESKHR> images;
  GLuint depth = 0;
  GLuint fbo = 0;
};

struct Hand {
  XrPath path = XR_NULL_PATH;
  XrSpace aimSpace = XR_NULL_HANDLE;
  bool active = false;
  Pose aim;
  float squeeze = 0, trigger = 0;
  XrVector2f stick{0, 0};
  bool stickClickPressed = false, primaryPressed = false, secondaryPressed = false,
       menuPressed = false;

  // Interaction state.
  bool grabbing = false;
  Pose grabOffset;
  float grabYaw = 0;
  bool triggerDown = false;
  int hoverButton = -1;
  int pressedButton = -1;
  float buttonRepeat = 0;
  int stickDir = 0;  // 0 none, 1 right, 2 left, 3 up, 4 down
  float stickRepeat = 0;

  // Hand tracking, used when this side's controller isn't active.
  XrHandTrackerEXT tracker = XR_NULL_HANDLE;
  bool tracked = false;
  XrHandJointLocationEXT joints[XR_HAND_JOINT_COUNT_EXT] = {};
  bool aimValid = false;       // Meta's hand "aim" ray is usable this frame
  float pinch = 0;             // index pinch strength, 0..1
  bool menuGesture = false;    // system menu gesture started this frame
  bool menuGestureDown = false;
  bool pinchDown = false;
  bool pinchOnSet = false;     // pinch began on the set body: a tap or the start of a grab
  float pinchHeld = 0;
  Vec3 pinchStart;
  bool nearSet = false;        // fingertip close enough to the set to poke instead of point
  int pokeButton = -1;
  bool pokeArmed = false;      // tip was in front of a button face, so pushing in counts
  std::vector<Pose> bindPoses;  // hand mesh bind pose per joint, when the runtime provides one

  // How long the controller pose has been reported valid but not actually tracked (e.g. it was
  // put down or lost); past a short grace period we stop treating it as present.
  float untracked = 0;
  bool loggedActive = false, loggedTracked = false;
};

class XrApp {
 public:
  bool run(JNIEnv* env, jobject activity, jobject bridge, jfloatArray initialPose,
           const char* modelPath, jfloatArray settings);

 private:
  Bridge bridge_;
  XrSettings settings_;

  // EGL
  EGLDisplay display_ = EGL_NO_DISPLAY;
  EGLConfig config_ = nullptr;
  EGLContext context_ = EGL_NO_CONTEXT;
  EGLSurface pbuffer_ = EGL_NO_SURFACE;
  bool msaa_ = false;
  PFNGLFRAMEBUFFERTEXTURE2DMULTISAMPLEEXTPROC glFramebufferTexture2DMultisampleEXT_ = nullptr;
  PFNGLRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC glRenderbufferStorageMultisampleEXT_ = nullptr;

  // OpenXR
  XrInstance instance_ = XR_NULL_HANDLE;
  XrSystemId system_ = XR_NULL_SYSTEM_ID;
  XrSession session_ = XR_NULL_HANDLE;
  XrSessionState state_ = XR_SESSION_STATE_UNKNOWN;
  bool running_ = false;
  bool exitRequestSent_ = false;
  XrSpace appSpace_ = XR_NULL_HANDLE;
  XrSpace viewSpace_ = XR_NULL_HANDLE;
  bool stageSpace_ = false;
  EyeTarget eyes_[2];
  int64_t colorFormat_ = GL_RGBA8;

  XrActionSet actionSet_ = XR_NULL_HANDLE;
  XrAction aimAction_ = XR_NULL_HANDLE, squeezeAction_ = XR_NULL_HANDLE,
           triggerAction_ = XR_NULL_HANDLE, stickAction_ = XR_NULL_HANDLE,
           stickClickAction_ = XR_NULL_HANDLE, primaryAction_ = XR_NULL_HANDLE,
           secondaryAction_ = XR_NULL_HANDLE, menuAction_ = XR_NULL_HANDLE,
           hapticAction_ = XR_NULL_HANDLE;
  Hand hands_[2];

  bool passthroughSupported_ = false;
  XrPassthroughFB passthrough_ = XR_NULL_HANDLE;
  XrPassthroughLayerFB passthroughLayer_ = XR_NULL_HANDLE;
  PFN_xrCreatePassthroughFB xrCreatePassthroughFB_ = nullptr;
  PFN_xrDestroyPassthroughFB xrDestroyPassthroughFB_ = nullptr;
  PFN_xrCreatePassthroughLayerFB xrCreatePassthroughLayerFB_ = nullptr;
  PFN_xrDestroyPassthroughLayerFB xrDestroyPassthroughLayerFB_ = nullptr;
  PFN_xrPassthroughLayerSetStyleFB xrPassthroughLayerSetStyleFB_ = nullptr;

  bool handTrackingSupported_ = false;
  bool handAimSupported_ = false;
  bool handMeshSupported_ = false;
  PFN_xrGetHandMeshFB xrGetHandMeshFB_ = nullptr;
  PFN_xrCreateHandTrackerEXT xrCreateHandTrackerEXT_ = nullptr;
  PFN_xrDestroyHandTrackerEXT xrDestroyHandTrackerEXT_ = nullptr;
  PFN_xrLocateHandJointsEXT xrLocateHandJointsEXT_ = nullptr;

  // Scene
  CrtScene scene_;
  GLuint videoTexture_ = 0;
  float texMatrix_[16] = {};
  Pose crtPose_;
  float crtScale_ = 1.0f;
  bool crtPlaced_ = false;
  bool haveHead_ = false;
  Pose head_;
  XrTime lastTime_ = 0;

  bool initEgl();
  bool initInstance(JNIEnv* env, jobject activity);
  bool initSession();
  bool initSwapchains();
  bool initActions();
  void initPassthrough();
  void initHandTracking();
  void shutdown();

  void pollEvents(bool& quit);
  void frame();
  void updateInput(XrTime time, float dt);
  void updateHand(int h, float dt);
  void updateTrackedHand(int h, float dt);
  void locateHand(int h, XrTime time);
  void startGrab(int h);
  void endGrab(int h);
  void pressButton(int h, int button);
  void repeatHeldButton(int h, float dt);
  int collectOccluders(int h, float* spheres, int max) const;
  void loadHandMesh(int h);
  // Puts the set `distance` metres ahead of the viewer, `drop` metres below eye level.
  void placeInFront(float distance = 1.4f, float drop = 0.05f);
  void haptic(int h, float amplitude);
  void renderEye(int eye, const XrView& view, uint32_t imageIndex);
};

bool XrApp::initEgl() {
  display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (!eglInitialize(display_, nullptr, nullptr)) {
    LOGE("eglInitialize failed");
    return false;
  }
  const EGLint configAttribs[] = {EGL_RED_SIZE,        8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                                  EGL_ALPHA_SIZE,      8, EGL_DEPTH_SIZE, 0, EGL_RENDERABLE_TYPE,
                                  EGL_OPENGL_ES3_BIT_KHR, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                  EGL_NONE};
  EGLint count = 0;
  if (!eglChooseConfig(display_, configAttribs, &config_, 1, &count) || count == 0) {
    LOGE("eglChooseConfig failed");
    return false;
  }
  const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
  context_ = eglCreateContext(display_, config_, EGL_NO_CONTEXT, contextAttribs);
  if (context_ == EGL_NO_CONTEXT) {
    LOGE("eglCreateContext failed");
    return false;
  }
  const EGLint pbufferAttribs[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
  pbuffer_ = eglCreatePbufferSurface(display_, config_, pbufferAttribs);
  if (!eglMakeCurrent(display_, pbuffer_, pbuffer_, context_)) {
    LOGE("eglMakeCurrent failed");
    return false;
  }

  const char* ext = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
  if (ext && strstr(ext, "GL_EXT_multisampled_render_to_texture")) {
    glFramebufferTexture2DMultisampleEXT_ =
        reinterpret_cast<PFNGLFRAMEBUFFERTEXTURE2DMULTISAMPLEEXTPROC>(
            eglGetProcAddress("glFramebufferTexture2DMultisampleEXT"));
    glRenderbufferStorageMultisampleEXT_ =
        reinterpret_cast<PFNGLRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC>(
            eglGetProcAddress("glRenderbufferStorageMultisampleEXT"));
    msaa_ = glFramebufferTexture2DMultisampleEXT_ && glRenderbufferStorageMultisampleEXT_;
  }
  LOGI("EGL ready, MSAA %s", msaa_ ? "on" : "off");
  return true;
}

bool XrApp::initInstance(JNIEnv* env, jobject activity) {
  JavaVM* vm = nullptr;
  env->GetJavaVM(&vm);

  PFN_xrInitializeLoaderKHR initializeLoader = nullptr;
  XR_CHECK(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR",
                                 reinterpret_cast<PFN_xrVoidFunction*>(&initializeLoader)));
  XrLoaderInitInfoAndroidKHR loaderInfo{XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
  loaderInfo.applicationVM = vm;
  loaderInfo.applicationContext = activity;
  XR_CHECK(initializeLoader(reinterpret_cast<XrLoaderInitInfoBaseHeaderKHR*>(&loaderInfo)));

  uint32_t extCount = 0;
  XR_CHECK(xrEnumerateInstanceExtensionProperties(nullptr, 0, &extCount, nullptr));
  std::vector<XrExtensionProperties> available(extCount, {XR_TYPE_EXTENSION_PROPERTIES});
  XR_CHECK(xrEnumerateInstanceExtensionProperties(nullptr, extCount, &extCount, available.data()));
  auto has = [&](const char* name) {
    return std::any_of(available.begin(), available.end(),
                       [&](const XrExtensionProperties& p) { return !strcmp(p.extensionName, name); });
  };

  std::vector<const char*> extensions = {XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME,
                                         XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME};
  for (const char* e : extensions) {
    if (!has(e)) {
      LOGE("required extension %s missing", e);
      return false;
    }
  }
  passthroughSupported_ = has(XR_FB_PASSTHROUGH_EXTENSION_NAME);
  if (passthroughSupported_) extensions.push_back(XR_FB_PASSTHROUGH_EXTENSION_NAME);
  handTrackingSupported_ = has(XR_EXT_HAND_TRACKING_EXTENSION_NAME);
  if (handTrackingSupported_) extensions.push_back(XR_EXT_HAND_TRACKING_EXTENSION_NAME);
  handAimSupported_ = handTrackingSupported_ && has(XR_FB_HAND_TRACKING_AIM_EXTENSION_NAME);
  if (handAimSupported_) extensions.push_back(XR_FB_HAND_TRACKING_AIM_EXTENSION_NAME);
  handMeshSupported_ = handTrackingSupported_ && has(XR_FB_HAND_TRACKING_MESH_EXTENSION_NAME);
  if (handMeshSupported_) extensions.push_back(XR_FB_HAND_TRACKING_MESH_EXTENSION_NAME);

  XrInstanceCreateInfoAndroidKHR androidInfo{XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
  androidInfo.applicationVM = vm;
  androidInfo.applicationActivity = activity;

  XrInstanceCreateInfo createInfo{XR_TYPE_INSTANCE_CREATE_INFO};
  createInfo.next = &androidInfo;
  strncpy(createInfo.applicationInfo.applicationName, "mpvEx", XR_MAX_APPLICATION_NAME_SIZE - 1);
  strncpy(createInfo.applicationInfo.engineName, "mpvEx", XR_MAX_ENGINE_NAME_SIZE - 1);
  createInfo.applicationInfo.applicationVersion = 1;
  createInfo.applicationInfo.apiVersion = XR_API_VERSION_1_0;
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.enabledExtensionNames = extensions.data();
  XR_CHECK(xrCreateInstance(&createInfo, &instance_));

  XrSystemGetInfo systemInfo{XR_TYPE_SYSTEM_GET_INFO};
  systemInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
  XR_CHECK(xrGetSystem(instance_, &systemInfo, &system_));

  PFN_xrGetOpenGLESGraphicsRequirementsKHR getRequirements = nullptr;
  XR_CHECK(xrGetInstanceProcAddr(instance_, "xrGetOpenGLESGraphicsRequirementsKHR",
                                 reinterpret_cast<PFN_xrVoidFunction*>(&getRequirements)));
  XrGraphicsRequirementsOpenGLESKHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR};
  XR_CHECK(getRequirements(instance_, system_, &requirements));

  if (passthroughSupported_) {
    auto load = [&](const char* name, auto& fn) {
      xrGetInstanceProcAddr(instance_, name, reinterpret_cast<PFN_xrVoidFunction*>(&fn));
    };
    load("xrCreatePassthroughFB", xrCreatePassthroughFB_);
    load("xrDestroyPassthroughFB", xrDestroyPassthroughFB_);
    load("xrCreatePassthroughLayerFB", xrCreatePassthroughLayerFB_);
    load("xrDestroyPassthroughLayerFB", xrDestroyPassthroughLayerFB_);
    load("xrPassthroughLayerSetStyleFB", xrPassthroughLayerSetStyleFB_);
    passthroughSupported_ = xrCreatePassthroughFB_ && xrDestroyPassthroughFB_ &&
                            xrCreatePassthroughLayerFB_ && xrDestroyPassthroughLayerFB_;
  }
  if (handTrackingSupported_) {
    XrSystemHandTrackingPropertiesEXT handProps{XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT};
    XrSystemProperties props{XR_TYPE_SYSTEM_PROPERTIES};
    props.next = &handProps;
    xrGetSystemProperties(instance_, system_, &props);
    auto load = [&](const char* name, auto& fn) {
      xrGetInstanceProcAddr(instance_, name, reinterpret_cast<PFN_xrVoidFunction*>(&fn));
    };
    load("xrCreateHandTrackerEXT", xrCreateHandTrackerEXT_);
    load("xrDestroyHandTrackerEXT", xrDestroyHandTrackerEXT_);
    load("xrLocateHandJointsEXT", xrLocateHandJointsEXT_);
    if (handMeshSupported_) load("xrGetHandMeshFB", xrGetHandMeshFB_);
    handMeshSupported_ = handMeshSupported_ && xrGetHandMeshFB_;
    handTrackingSupported_ = handProps.supportsHandTracking && xrCreateHandTrackerEXT_ &&
                             xrDestroyHandTrackerEXT_ && xrLocateHandJointsEXT_;
  }
  LOGI("OpenXR instance ready, passthrough %s, hand tracking %s (aim %s, mesh %s)",
       passthroughSupported_ ? "available" : "missing", handTrackingSupported_ ? "yes" : "no",
       handAimSupported_ ? "yes" : "no", handMeshSupported_ ? "yes" : "no");
  return true;
}

bool XrApp::initSession() {
  XrGraphicsBindingOpenGLESAndroidKHR binding{XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR};
  binding.display = display_;
  binding.config = config_;
  binding.context = context_;
  XrSessionCreateInfo createInfo{XR_TYPE_SESSION_CREATE_INFO};
  createInfo.next = &binding;
  createInfo.systemId = system_;
  XR_CHECK(xrCreateSession(instance_, &createInfo, &session_));

  uint32_t spaceCount = 0;
  XR_CHECK(xrEnumerateReferenceSpaces(session_, 0, &spaceCount, nullptr));
  std::vector<XrReferenceSpaceType> spaces(spaceCount);
  XR_CHECK(xrEnumerateReferenceSpaces(session_, spaceCount, &spaceCount, spaces.data()));
  stageSpace_ = std::find(spaces.begin(), spaces.end(), XR_REFERENCE_SPACE_TYPE_STAGE) != spaces.end();

  XrReferenceSpaceCreateInfo spaceInfo{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  spaceInfo.poseInReferenceSpace.orientation.w = 1.0f;
  spaceInfo.referenceSpaceType = stageSpace_ ? XR_REFERENCE_SPACE_TYPE_STAGE : XR_REFERENCE_SPACE_TYPE_LOCAL;
  XR_CHECK(xrCreateReferenceSpace(session_, &spaceInfo, &appSpace_));
  spaceInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
  XR_CHECK(xrCreateReferenceSpace(session_, &spaceInfo, &viewSpace_));
  return true;
}

bool XrApp::initSwapchains() {
  uint32_t viewCount = 0;
  XR_CHECK(xrEnumerateViewConfigurationViews(instance_, system_,
                                             XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0,
                                             &viewCount, nullptr));
  std::vector<XrViewConfigurationView> views(viewCount, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
  XR_CHECK(xrEnumerateViewConfigurationViews(instance_, system_,
                                             XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount,
                                             &viewCount, views.data()));
  if (viewCount != 2) {
    LOGE("expected 2 views, got %u", viewCount);
    return false;
  }

  uint32_t formatCount = 0;
  XR_CHECK(xrEnumerateSwapchainFormats(session_, 0, &formatCount, nullptr));
  std::vector<int64_t> formats(formatCount);
  XR_CHECK(xrEnumerateSwapchainFormats(session_, formatCount, &formatCount, formats.data()));
  // sRGB lets the compositor treat our output correctly; the shaders work in linear space.
  colorFormat_ = std::find(formats.begin(), formats.end(), GL_SRGB8_ALPHA8) != formats.end()
                     ? GL_SRGB8_ALPHA8
                     : GL_RGBA8;

  for (int i = 0; i < 2; ++i) {
    EyeTarget& eye = eyes_[i];
    eye.width = static_cast<int32_t>(views[i].recommendedImageRectWidth);
    eye.height = static_cast<int32_t>(views[i].recommendedImageRectHeight);

    XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    info.format = colorFormat_;
    info.sampleCount = 1;
    info.width = eye.width;
    info.height = eye.height;
    info.faceCount = 1;
    info.arraySize = 1;
    info.mipCount = 1;
    XR_CHECK(xrCreateSwapchain(session_, &info, &eye.swapchain));

    uint32_t imageCount = 0;
    XR_CHECK(xrEnumerateSwapchainImages(eye.swapchain, 0, &imageCount, nullptr));
    eye.images.assign(imageCount, {XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR});
    XR_CHECK(xrEnumerateSwapchainImages(
        eye.swapchain, imageCount, &imageCount,
        reinterpret_cast<XrSwapchainImageBaseHeader*>(eye.images.data())));

    glGenRenderbuffers(1, &eye.depth);
    glBindRenderbuffer(GL_RENDERBUFFER, eye.depth);
    if (msaa_) {
      glRenderbufferStorageMultisampleEXT_(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT24, eye.width,
                                           eye.height);
    } else {
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, eye.width, eye.height);
    }
    glGenFramebuffers(1, &eye.fbo);
  }
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  LOGI("swapchains %dx%d format 0x%llx", eyes_[0].width, eyes_[0].height,
       static_cast<long long>(colorFormat_));
  return true;
}

bool XrApp::initActions() {
  XrActionSetCreateInfo setInfo{XR_TYPE_ACTION_SET_CREATE_INFO};
  strcpy(setInfo.actionSetName, "player");
  strcpy(setInfo.localizedActionSetName, "Player");
  XR_CHECK(xrCreateActionSet(instance_, &setInfo, &actionSet_));

  XR_CHECK(xrStringToPath(instance_, "/user/hand/left", &hands_[kLeft].path));
  XR_CHECK(xrStringToPath(instance_, "/user/hand/right", &hands_[kRight].path));
  XrPath handPaths[2] = {hands_[kLeft].path, hands_[kRight].path};

  auto create = [&](XrAction& action, XrActionType type, const char* name, const char* label) {
    XrActionCreateInfo info{XR_TYPE_ACTION_CREATE_INFO};
    info.actionType = type;
    strcpy(info.actionName, name);
    strcpy(info.localizedActionName, label);
    info.countSubactionPaths = 2;
    info.subactionPaths = handPaths;
    return xrCreateAction(actionSet_, &info, &action);
  };
  XR_CHECK(create(aimAction_, XR_ACTION_TYPE_POSE_INPUT, "aim", "Aim"));
  XR_CHECK(create(squeezeAction_, XR_ACTION_TYPE_FLOAT_INPUT, "grab", "Grab"));
  XR_CHECK(create(triggerAction_, XR_ACTION_TYPE_FLOAT_INPUT, "select", "Select"));
  XR_CHECK(create(stickAction_, XR_ACTION_TYPE_VECTOR2F_INPUT, "thumbstick", "Thumbstick"));
  XR_CHECK(create(stickClickAction_, XR_ACTION_TYPE_BOOLEAN_INPUT, "thumbstick_click", "Thumbstick click"));
  XR_CHECK(create(primaryAction_, XR_ACTION_TYPE_BOOLEAN_INPUT, "primary", "Primary button"));
  XR_CHECK(create(secondaryAction_, XR_ACTION_TYPE_BOOLEAN_INPUT, "secondary", "Secondary button"));
  XR_CHECK(create(menuAction_, XR_ACTION_TYPE_BOOLEAN_INPUT, "menu", "Menu"));
  XR_CHECK(create(hapticAction_, XR_ACTION_TYPE_VIBRATION_OUTPUT, "haptic", "Haptic"));

  std::vector<XrActionSuggestedBinding> bindings;
  auto bind = [&](XrAction action, const char* path) {
    XrPath p;
    if (XR_SUCCEEDED(xrStringToPath(instance_, path, &p))) bindings.push_back({action, p});
  };
  for (const char* side : {"left", "right"}) {
    std::string base = std::string("/user/hand/") + side;
    bind(aimAction_, (base + "/input/aim/pose").c_str());
    bind(squeezeAction_, (base + "/input/squeeze/value").c_str());
    bind(triggerAction_, (base + "/input/trigger/value").c_str());
    bind(stickAction_, (base + "/input/thumbstick").c_str());
    bind(stickClickAction_, (base + "/input/thumbstick/click").c_str());
    bind(hapticAction_, (base + "/output/haptic").c_str());
  }
  bind(primaryAction_, "/user/hand/left/input/x/click");
  bind(primaryAction_, "/user/hand/right/input/a/click");
  bind(secondaryAction_, "/user/hand/left/input/y/click");
  bind(secondaryAction_, "/user/hand/right/input/b/click");
  bind(menuAction_, "/user/hand/left/input/menu/click");

  XrPath profile;
  XR_CHECK(xrStringToPath(instance_, "/interaction_profiles/oculus/touch_controller", &profile));
  XrInteractionProfileSuggestedBinding suggested{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
  suggested.interactionProfile = profile;
  suggested.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
  suggested.suggestedBindings = bindings.data();
  XR_CHECK(xrSuggestInteractionProfileBindings(instance_, &suggested));

  for (Hand& hand : hands_) {
    XrActionSpaceCreateInfo spaceInfo{XR_TYPE_ACTION_SPACE_CREATE_INFO};
    spaceInfo.action = aimAction_;
    spaceInfo.subactionPath = hand.path;
    spaceInfo.poseInActionSpace.orientation.w = 1.0f;
    XR_CHECK(xrCreateActionSpace(session_, &spaceInfo, &hand.aimSpace));
  }

  XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
  attach.countActionSets = 1;
  attach.actionSets = &actionSet_;
  XR_CHECK(xrAttachSessionActionSets(session_, &attach));
  return true;
}

void XrApp::initPassthrough() {
  if (!passthroughSupported_) return;
  XrPassthroughCreateInfoFB info{XR_TYPE_PASSTHROUGH_CREATE_INFO_FB};
  info.flags = XR_PASSTHROUGH_IS_RUNNING_AT_CREATION_BIT_FB;
  if (XR_FAILED(xrCreatePassthroughFB_(session_, &info, &passthrough_))) {
    LOGE("xrCreatePassthroughFB failed");
    passthroughSupported_ = false;
    return;
  }
  XrPassthroughLayerCreateInfoFB layerInfo{XR_TYPE_PASSTHROUGH_LAYER_CREATE_INFO_FB};
  layerInfo.passthrough = passthrough_;
  layerInfo.flags = XR_PASSTHROUGH_IS_RUNNING_AT_CREATION_BIT_FB;
  layerInfo.purpose = XR_PASSTHROUGH_LAYER_PURPOSE_RECONSTRUCTION_FB;
  if (XR_FAILED(xrCreatePassthroughLayerFB_(session_, &layerInfo, &passthroughLayer_))) {
    LOGE("xrCreatePassthroughLayerFB failed");
    xrDestroyPassthroughFB_(passthrough_);
    passthrough_ = XR_NULL_HANDLE;
    passthroughSupported_ = false;
    return;
  }

  // Room dimming fades passthrough towards black; saturation greys it out.
  if (xrPassthroughLayerSetStyleFB_ &&
      (settings_.passthroughOpacity < 1.0f || settings_.passthroughSaturation < 1.0f)) {
    XrPassthroughBrightnessContrastSaturationFB bcs{XR_TYPE_PASSTHROUGH_BRIGHTNESS_CONTRAST_SATURATION_FB};
    bcs.brightness = 0.0f;
    bcs.contrast = 1.0f;
    bcs.saturation = settings_.passthroughSaturation;
    XrPassthroughStyleFB style{XR_TYPE_PASSTHROUGH_STYLE_FB};
    style.next = &bcs;
    style.textureOpacityFactor = settings_.passthroughOpacity;
    style.edgeColor = {0, 0, 0, 0};
    if (XR_FAILED(xrPassthroughLayerSetStyleFB_(passthroughLayer_, &style))) {
      LOGE("xrPassthroughLayerSetStyleFB failed");
    }
  }
}

void XrApp::initHandTracking() {
  if (!handTrackingSupported_) return;
  const XrHandEXT sides[2] = {XR_HAND_LEFT_EXT, XR_HAND_RIGHT_EXT};
  for (int h = 0; h < 2; ++h) {
    XrHandTrackerCreateInfoEXT info{XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT};
    info.hand = sides[h];
    info.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;
    if (XR_FAILED(xrCreateHandTrackerEXT_(session_, &info, &hands_[h].tracker))) {
      LOGE("xrCreateHandTrackerEXT failed");
      hands_[h].tracker = XR_NULL_HANDLE;
      continue;
    }
    loadHandMesh(h);
  }
}

void XrApp::loadHandMesh(int h) {
  Hand& hand = hands_[h];
  if (!handMeshSupported_ || !hand.tracker) return;
  XrHandTrackingMeshFB mesh{XR_TYPE_HAND_TRACKING_MESH_FB};
  if (XR_FAILED(xrGetHandMeshFB_(hand.tracker, &mesh))) return;
  if (mesh.jointCountOutput != CrtScene::kHandJoints) {
    LOGE("hand mesh has %u joints, expected %d", mesh.jointCountOutput, CrtScene::kHandJoints);
    return;
  }
  std::vector<XrPosef> bind(mesh.jointCountOutput);
  std::vector<float> radii(mesh.jointCountOutput);
  std::vector<XrHandJointEXT> parents(mesh.jointCountOutput);
  std::vector<XrVector3f> positions(mesh.vertexCountOutput), normals(mesh.vertexCountOutput);
  std::vector<XrVector2f> uvs(mesh.vertexCountOutput);
  std::vector<XrVector4sFB> joints(mesh.vertexCountOutput);
  std::vector<XrVector4f> weights(mesh.vertexCountOutput);
  std::vector<int16_t> indices(mesh.indexCountOutput);
  mesh.jointCapacityInput = mesh.jointCountOutput;
  mesh.jointBindPoses = bind.data();
  mesh.jointRadii = radii.data();
  mesh.jointParents = parents.data();
  mesh.vertexCapacityInput = mesh.vertexCountOutput;
  mesh.vertexPositions = positions.data();
  mesh.vertexNormals = normals.data();
  mesh.vertexUVs = uvs.data();
  mesh.vertexBlendIndices = joints.data();
  mesh.vertexBlendWeights = weights.data();
  mesh.indexCapacityInput = mesh.indexCountOutput;
  mesh.indices = indices.data();
  if (XR_FAILED(xrGetHandMeshFB_(hand.tracker, &mesh))) return;

  hand.bindPoses.clear();
  for (const XrPosef& p : bind) hand.bindPoses.push_back(fromXr(p));
  static_assert(sizeof(XrVector3f) == 3 * sizeof(float), "packed");
  static_assert(sizeof(XrVector4sFB) == 4 * sizeof(int16_t), "packed");
  static_assert(sizeof(XrVector4f) == 4 * sizeof(float), "packed");
  scene_.setHandMesh(h, &positions[0].x, &normals[0].x, &joints[0].x, &weights[0].x,
                     static_cast<int>(mesh.vertexCountOutput), indices.data(),
                     static_cast<int>(mesh.indexCountOutput));
  LOGI("hand %d mesh: %u vertices, %u triangles", h, mesh.vertexCountOutput,
       mesh.indexCountOutput / 3);
}

void XrApp::shutdown() {
  for (Hand& hand : hands_) {
    if (hand.tracker) xrDestroyHandTrackerEXT_(hand.tracker);
    hand.tracker = XR_NULL_HANDLE;
  }
  for (EyeTarget& eye : eyes_) {
    if (eye.fbo) glDeleteFramebuffers(1, &eye.fbo);
    if (eye.depth) glDeleteRenderbuffers(1, &eye.depth);
    if (eye.swapchain) xrDestroySwapchain(eye.swapchain);
    eye = {};
  }
  if (passthroughLayer_) xrDestroyPassthroughLayerFB_(passthroughLayer_);
  if (passthrough_) xrDestroyPassthroughFB_(passthrough_);
  for (Hand& hand : hands_) {
    if (hand.aimSpace) xrDestroySpace(hand.aimSpace);
  }
  if (actionSet_) xrDestroyActionSet(actionSet_);
  if (viewSpace_) xrDestroySpace(viewSpace_);
  if (appSpace_) xrDestroySpace(appSpace_);
  if (session_) xrDestroySession(session_);
  if (instance_) xrDestroyInstance(instance_);

  scene_.destroy();
  if (videoTexture_) glDeleteTextures(1, &videoTexture_);

  if (display_ != EGL_NO_DISPLAY) {
    eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (pbuffer_ != EGL_NO_SURFACE) eglDestroySurface(display_, pbuffer_);
    if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
    eglTerminate(display_);
  }
}

void XrApp::pollEvents(bool& quit) {
  XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
  while (xrPollEvent(instance_, &event) == XR_SUCCESS) {
    switch (event.type) {
      case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
        quit = true;
        break;
      case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
        auto* e = reinterpret_cast<XrEventDataSessionStateChanged*>(&event);
        state_ = e->state;
        LOGI("session state %d", state_);
        if (state_ == XR_SESSION_STATE_READY) {
          XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO};
          begin.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
          if (XR_SUCCEEDED(xrBeginSession(session_, &begin))) running_ = true;
        } else if (state_ == XR_SESSION_STATE_STOPPING) {
          xrEndSession(session_);
          running_ = false;
          bridge_.action(kPause);
        } else if (state_ == XR_SESSION_STATE_EXITING || state_ == XR_SESSION_STATE_LOSS_PENDING) {
          quit = true;
        }
        break;
      }
      default:
        break;
    }
    event = {XR_TYPE_EVENT_DATA_BUFFER};
  }
}

void XrApp::haptic(int h, float amplitude) {
  XrHapticVibration vibration{XR_TYPE_HAPTIC_VIBRATION};
  vibration.duration = 20'000'000;  // 20 ms
  vibration.frequency = XR_FREQUENCY_UNSPECIFIED;
  vibration.amplitude = amplitude;
  XrHapticActionInfo info{XR_TYPE_HAPTIC_ACTION_INFO};
  info.action = hapticAction_;
  info.subactionPath = hands_[h].path;
  xrApplyHapticFeedback(session_, &info, reinterpret_cast<XrHapticBaseHeader*>(&vibration));
}

void XrApp::placeInFront(float distance, float drop) {
  if (!haveHead_) return;
  float yaw = yawOf(head_.q);
  Vec3 forward{-std::sin(yaw), 0, -std::cos(yaw)};
  crtPose_.p = head_.p + forward * distance;
  crtPose_.p.y = head_.p.y - drop;
  // The screen faces +Z, so the same yaw as the head points it back at the viewer.
  crtPose_.q = axisAngle({0, 1, 0}, yaw);
  crtPlaced_ = true;
  bridge_.crtPoseChanged(crtPose_, crtScale_);
}

void XrApp::updateInput(XrTime time, float dt) {
  XrSpaceLocation headLocation{XR_TYPE_SPACE_LOCATION};
  if (XR_SUCCEEDED(xrLocateSpace(viewSpace_, appSpace_, time, &headLocation)) &&
      (headLocation.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
      (headLocation.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
    head_ = fromXr(headLocation.pose);
    haveHead_ = true;
  }
  if (!crtPlaced_) placeInFront();

  bool focused = state_ == XR_SESSION_STATE_FOCUSED;
  if (focused) {
    XrActiveActionSet active{actionSet_, XR_NULL_PATH};
    XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO};
    sync.countActiveActionSets = 1;
    sync.activeActionSets = &active;
    focused = XR_SUCCEEDED(xrSyncActions(session_, &sync));
  }

  for (int h = 0; h < 2; ++h) {
    Hand& hand = hands_[h];
    hand.active = false;
    if (!focused) continue;

    XrActionStateGetInfo get{XR_TYPE_ACTION_STATE_GET_INFO};
    get.subactionPath = hand.path;

    get.action = aimAction_;
    XrActionStatePose poseState{XR_TYPE_ACTION_STATE_POSE};
    xrGetActionStatePose(session_, &get, &poseState);
    XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
    constexpr XrSpaceLocationFlags kValid =
        XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
    constexpr XrSpaceLocationFlags kTracked =
        XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
    if (poseState.isActive && XR_SUCCEEDED(xrLocateSpace(hand.aimSpace, appSpace_, time, &location)) &&
        (location.locationFlags & kValid) == kValid) {
      hand.untracked = (location.locationFlags & kTracked) == kTracked ? 0.0f : hand.untracked + dt;
      if (hand.untracked < 0.3f) {
        hand.aim = fromXr(location.pose);
        hand.active = true;
      }
    } else {
      hand.untracked = 0;
    }

    auto getFloat = [&](XrAction action) {
      get.action = action;
      XrActionStateFloat s{XR_TYPE_ACTION_STATE_FLOAT};
      xrGetActionStateFloat(session_, &get, &s);
      return s.isActive ? s.currentState : 0.0f;
    };
    auto pressed = [&](XrAction action) {
      get.action = action;
      XrActionStateBoolean s{XR_TYPE_ACTION_STATE_BOOLEAN};
      xrGetActionStateBoolean(session_, &get, &s);
      return s.isActive && s.changedSinceLastSync && s.currentState;
    };
    hand.squeeze = getFloat(squeezeAction_);
    hand.trigger = getFloat(triggerAction_);
    get.action = stickAction_;
    XrActionStateVector2f stick{XR_TYPE_ACTION_STATE_VECTOR2F};
    xrGetActionStateVector2f(session_, &get, &stick);
    hand.stick = stick.isActive ? stick.currentState : XrVector2f{0, 0};
    hand.stickClickPressed = pressed(stickClickAction_);
    hand.primaryPressed = pressed(primaryAction_);
    hand.secondaryPressed = pressed(secondaryAction_);
    hand.menuPressed = pressed(menuAction_);
  }

  for (int h = 0; h < 2; ++h) {
    if (focused && !hands_[h].active) {
      locateHand(h, time);
    } else {
      hands_[h].tracked = false;
    }
  }

  for (int h = 0; h < 2; ++h) {
    Hand& hand = hands_[h];
    const char* side = h == kLeft ? "left" : "right";
    if (hand.active != hand.loggedActive) LOGI("%s controller %s", side, hand.active ? "active" : "gone");
    if (hand.tracked != hand.loggedTracked) LOGI("%s hand %s", side, hand.tracked ? "tracked" : "lost");
    hand.loggedActive = hand.active;
    hand.loggedTracked = hand.tracked;
  }

  for (int h = 0; h < 2; ++h) updateHand(h, dt);
}

void XrApp::locateHand(int h, XrTime time) {
  Hand& hand = hands_[h];
  hand.tracked = false;
  hand.aimValid = false;
  hand.menuGesture = false;
  if (!hand.tracker) return;

  XrHandTrackingAimStateFB aim{XR_TYPE_HAND_TRACKING_AIM_STATE_FB};
  XrHandJointLocationsEXT locations{XR_TYPE_HAND_JOINT_LOCATIONS_EXT};
  locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
  locations.jointLocations = hand.joints;
  if (handAimSupported_) locations.next = &aim;
  XrHandJointsLocateInfoEXT info{XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT};
  info.baseSpace = appSpace_;
  info.time = time;
  if (XR_FAILED(xrLocateHandJointsEXT_(hand.tracker, &info, &locations)) || !locations.isActive) {
    return;
  }
  // A lost hand can still come back "active" with a held pose; only use it while really tracked.
  constexpr XrSpaceLocationFlags kTracked =
      XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
  if ((hand.joints[XR_HAND_JOINT_PALM_EXT].locationFlags & kTracked) != kTracked) return;
  hand.tracked = true;

  constexpr XrHandTrackingAimFlagsFB kAimUsable =
      XR_HAND_TRACKING_AIM_COMPUTED_BIT_FB | XR_HAND_TRACKING_AIM_VALID_BIT_FB;
  if (handAimSupported_ && (aim.status & kAimUsable) == kAimUsable) {
    hand.aimValid = true;
    hand.aim = fromXr(aim.aimPose);
    hand.pinch = aim.pinchStrengthIndex;
  } else {
    // No aim extension: point along the index finger and pinch by thumb-index distance.
    const XrPosef& palm = hand.joints[XR_HAND_JOINT_PALM_EXT].pose;
    const XrVector3f& t = hand.joints[XR_HAND_JOINT_THUMB_TIP_EXT].pose.position;
    const XrVector3f& i = hand.joints[XR_HAND_JOINT_INDEX_TIP_EXT].pose.position;
    hand.aimValid = hand.joints[XR_HAND_JOINT_PALM_EXT].locationFlags &
                    XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
    hand.aim = fromXr(palm);
    float gap = length(Vec3{t.x - i.x, t.y - i.y, t.z - i.z});
    hand.pinch = std::clamp(1.0f - (gap - 0.01f) / 0.03f, 0.0f, 1.0f);
  }
  bool menu = handAimSupported_ && (aim.status & XR_HAND_TRACKING_AIM_MENU_PRESSED_BIT_FB);
  hand.menuGesture = menu && !hand.menuGestureDown;
  hand.menuGestureDown = menu;
}

void XrApp::startGrab(int h) {
  Hand& hand = hands_[h];
  hand.grabbing = true;
  hand.pressedButton = -1;
  hand.grabOffset = inverse(hand.aim) * crtPose_;
  hand.grabYaw = 0;
  haptic(h, 0.5f);
}

void XrApp::endGrab(int h) {
  hands_[h].grabbing = false;
  haptic(h, 0.25f);
  bridge_.crtPoseChanged(crtPose_, crtScale_);
}

void XrApp::pressButton(int h, int button) {
  Hand& hand = hands_[h];
  hand.pressedButton = button;
  hand.buttonRepeat = 0.45f;
  haptic(h, 0.3f);
  bridge_.action(scene_.button(button).action);
}

void XrApp::repeatHeldButton(int h, float dt) {
  // Holding rewind / fast-forward / volume keeps going while still on the button.
  Hand& hand = hands_[h];
  const CrtButton& held = scene_.button(hand.pressedButton);
  if (!held.repeats || hand.hoverButton != hand.pressedButton) return;
  hand.buttonRepeat -= dt;
  if (hand.buttonRepeat <= 0) {
    hand.buttonRepeat = 0.3f;
    bridge_.action(held.action);
  }
}

void XrApp::updateTrackedHand(int h, float dt) {
  Hand& hand = hands_[h];
  Hand& other = hands_[1 - h];

  // Poke: the index fingertip presses the set's buttons directly.
  const XrHandJointLocationEXT& tipJoint = hand.joints[XR_HAND_JOINT_INDEX_TIP_EXT];
  bool tipValid = tipJoint.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT;
  Vec3 tip{tipJoint.pose.position.x, tipJoint.pose.position.y, tipJoint.pose.position.z};
  hand.nearSet = !hand.grabbing && tipValid && scene_.nearFront(crtPose_, crtScale_, tip, 0.12f);

  if (hand.nearSet) {
    // Leaving laser mode mid-pinch shouldn't leave a pending tap or held button behind.
    if (hand.pinchDown && hand.pressedButton >= 0 && hand.pokeButton < 0) hand.pressedButton = -1;
    hand.pinchDown = hand.pinchOnSet = false;

    float depth = 0;
    int b = scene_.buttonUnderPoint(crtPose_, crtScale_, tip, &depth);
    if (b >= 0 && depth > -0.03f) hand.hoverButton = b;
    if (hand.pokeButton >= 0) {
      // Release once the finger backs off the face or slides off the button.
      if (b != hand.pokeButton || depth < -0.004f) {
        hand.pokeButton = -1;
        hand.pressedButton = -1;
      } else {
        repeatHeldButton(h, dt);
      }
    } else if (b >= 0 && depth > 0 && hand.pokeArmed) {
      hand.pokeButton = b;
      pressButton(h, b);
    }
    // Only a press that starts in front of the face counts, so sliding sideways onto a button
    // while already pushed into the panel does nothing.
    if (b < 0) {
      hand.pokeArmed = false;
    } else if (depth < -0.002f) {
      hand.pokeArmed = true;
    }
    return;
  }
  if (hand.pokeButton >= 0) {
    hand.pokeButton = -1;
    hand.pressedButton = -1;
  }
  hand.pokeArmed = false;

  // Within arm's reach, so the buttons can be poked.
  if (hand.menuGesture) placeInFront(settings_.reachDistance, 0.25f);

  // Laser: pinch acts like the trigger, and a held pinch on the set grabs it.
  if (!hand.aimValid) {
    if (hand.grabbing) endGrab(h);
    hand.pinchDown = hand.pinchOnSet = false;
    hand.pressedButton = -1;
    return;
  }
  Vec3 origin = hand.aim.p;
  Vec3 dir = rotate(hand.aim.q, {0, 0, -1});
  float hit = scene_.rayHit(crtPose_, crtScale_, origin, dir);
  bool pointing = hit >= 0.0f && hit < 20.0f;
  if (pointing && !hand.grabbing) hand.hoverButton = scene_.buttonAt(crtPose_, crtScale_, origin, dir);

  bool pinchDown = hand.pinch > 0.85f || (hand.pinchDown && hand.pinch > 0.6f);
  if (pinchDown && !hand.pinchDown) {
    if (hand.hoverButton >= 0) {
      pressButton(h, hand.hoverButton);
    } else if (pointing) {
      hand.pinchOnSet = true;
      hand.pinchHeld = 0;
      hand.pinchStart = origin;
    }
  } else if (pinchDown) {
    if (hand.pressedButton >= 0) repeatHeldButton(h, dt);
    if (hand.pinchOnSet && !hand.grabbing && !other.grabbing) {
      hand.pinchHeld += dt;
      if (hand.pinchHeld > 0.35f || length(origin - hand.pinchStart) > 0.03f) startGrab(h);
    }
  } else if (hand.pinchDown) {
    if (hand.grabbing) {
      endGrab(h);
    } else if (hand.pinchOnSet) {
      bridge_.action(kTogglePause);
    }
    hand.pinchOnSet = false;
    hand.pressedButton = -1;
  }
  hand.pinchDown = pinchDown;

  if (hand.grabbing) {
    Pose held = hand.aim * hand.grabOffset;
    crtPose_.p = held.p;
    crtPose_.q = axisAngle({0, 1, 0}, yawOf(held.q));
  }
}

int XrApp::collectOccluders(int h, float* spheres, int max) const {
  // Spheres along every finger bone, slightly inflated to cover tracking error.
  static const int kChains[][6] = {
      {XR_HAND_JOINT_WRIST_EXT, XR_HAND_JOINT_THUMB_METACARPAL_EXT, XR_HAND_JOINT_THUMB_PROXIMAL_EXT,
       XR_HAND_JOINT_THUMB_DISTAL_EXT, XR_HAND_JOINT_THUMB_TIP_EXT, -1},
      {XR_HAND_JOINT_WRIST_EXT, XR_HAND_JOINT_INDEX_PROXIMAL_EXT, XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT,
       XR_HAND_JOINT_INDEX_DISTAL_EXT, XR_HAND_JOINT_INDEX_TIP_EXT, -1},
      {XR_HAND_JOINT_WRIST_EXT, XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT, XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT,
       XR_HAND_JOINT_MIDDLE_DISTAL_EXT, XR_HAND_JOINT_MIDDLE_TIP_EXT, -1},
      {XR_HAND_JOINT_WRIST_EXT, XR_HAND_JOINT_RING_PROXIMAL_EXT, XR_HAND_JOINT_RING_INTERMEDIATE_EXT,
       XR_HAND_JOINT_RING_DISTAL_EXT, XR_HAND_JOINT_RING_TIP_EXT, -1},
      {XR_HAND_JOINT_WRIST_EXT, XR_HAND_JOINT_LITTLE_PROXIMAL_EXT, XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT,
       XR_HAND_JOINT_LITTLE_DISTAL_EXT, XR_HAND_JOINT_LITTLE_TIP_EXT, -1},
  };
  int n = 0;
  auto add = [&](Vec3 p, float r) {
    if (n >= max) return;
    spheres[n * 4] = p.x;
    spheres[n * 4 + 1] = p.y;
    spheres[n * 4 + 2] = p.z;
    spheres[n * 4 + 3] = r;
    ++n;
  };
  {
    const Hand& hand = hands_[h];
    if (!hand.tracked) return 0;
    auto joint = [&](int j, Vec3* p, float* r) {
      const XrHandJointLocationEXT& l = hand.joints[j];
      if (!(l.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT)) return false;
      *p = {l.pose.position.x, l.pose.position.y, l.pose.position.z};
      *r = std::max(l.radius, 0.006f) * 1.1f + settings_.handPadding;
      return true;
    };
    Vec3 palm;
    float palmR;
    if (joint(XR_HAND_JOINT_PALM_EXT, &palm, &palmR)) add(palm, std::max(palmR, 0.035f));
    for (const auto& chain : kChains) {
      for (int k = 0; chain[k + 1] >= 0; ++k) {
        Vec3 a, b;
        float ra, rb;
        if (!joint(chain[k], &a, &ra) || !joint(chain[k + 1], &b, &rb)) continue;
        // Enough spheres along the bone that neighbours overlap.
        int steps = std::clamp(static_cast<int>(length(b - a) / (std::min(ra, rb) * 1.2f)), 1, 6);
        for (int s = 0; s < steps; ++s) {
          float t = static_cast<float>(s) / steps;
          add(a + (b - a) * t, ra + (rb - ra) * t);
        }
        if (chain[k + 2] < 0) add(b, rb);
      }
    }
  }
  return n;
}

void XrApp::updateHand(int h, float dt) {
  Hand& hand = hands_[h];
  Hand& other = hands_[1 - h];
  hand.hoverButton = -1;
  if (!hand.active) {
    if (hand.tracked) {
      updateTrackedHand(h, dt);
      return;
    }
    hand.pressedButton = -1;
    hand.pokeButton = -1;
    hand.pinchDown = hand.pinchOnSet = false;
    if (hand.grabbing) {
      hand.grabbing = false;
      bridge_.crtPoseChanged(crtPose_, crtScale_);
    }
    return;
  }

  Vec3 origin = hand.aim.p;
  Vec3 dir = rotate(hand.aim.q, {0, 0, -1});
  float hit = scene_.rayHit(crtPose_, crtScale_, origin, dir);
  bool pointing = hit >= 0.0f && hit < 20.0f;

  // Grab start / end with hysteresis.
  if (!hand.grabbing && !other.grabbing && pointing && hand.squeeze > 0.6f) {
    hand.grabbing = true;
    hand.pressedButton = -1;
    hand.grabOffset = inverse(hand.aim) * crtPose_;
    hand.grabYaw = 0;
    haptic(h, 0.5f);
  } else if (hand.grabbing && hand.squeeze < 0.4f) {
    hand.grabbing = false;
    haptic(h, 0.25f);
    bridge_.crtPoseChanged(crtPose_, crtScale_);
  }

  if (hand.grabbing) {
    // Thumbstick Y pushes the set away or pulls it in along the ray; X spins it.
    // The controller looks down -Z, so a more negative offset is further away.
    if (std::fabs(hand.stick.y) > 0.15f) {
      hand.grabOffset.p.z = std::min(hand.grabOffset.p.z - hand.stick.y * dt * 1.5f, -0.3f);
    }
    if (std::fabs(hand.stick.x) > 0.15f) hand.grabYaw -= hand.stick.x * dt * 2.0f;
    if (hand.stickClickPressed) {
      const float modelInches = scene_.screenDiagonalInches();
      float inches = crtScale_ * modelInches;
      float next = kSizePresetsInches[0];
      for (float preset : kSizePresetsInches) {
        if (preset > inches + 0.5f) {
          next = preset;
          break;
        }
      }
      crtScale_ = next / modelInches;
      haptic(h, 0.35f);
    }
    Pose held = hand.aim * hand.grabOffset;
    // Keep the set upright; only its heading follows the controller.
    crtPose_.p = held.p;
    crtPose_.q = axisAngle({0, 1, 0}, yawOf(held.q) + hand.grabYaw);
    return;
  }

  // Trigger presses front-panel buttons; anywhere else on the set toggles playback.
  if (pointing) hand.hoverButton = scene_.buttonAt(crtPose_, crtScale_, origin, dir);
  bool triggerDown = hand.trigger > 0.7f || (hand.triggerDown && hand.trigger > 0.5f);
  if (triggerDown && !hand.triggerDown) {
    if (hand.hoverButton >= 0) {
      hand.pressedButton = hand.hoverButton;
      hand.buttonRepeat = 0.45f;
      haptic(h, 0.3f);
      bridge_.action(scene_.button(hand.pressedButton).action);
    } else if (pointing) {
      bridge_.action(kTogglePause);
    }
  } else if (triggerDown && hand.pressedButton >= 0) {
    // Holding rewind / fast-forward / volume keeps going while still on the button.
    const CrtButton& held = scene_.button(hand.pressedButton);
    if (held.repeats && hand.hoverButton == hand.pressedButton) {
      hand.buttonRepeat -= dt;
      if (hand.buttonRepeat <= 0) {
        hand.buttonRepeat = 0.3f;
        bridge_.action(held.action);
      }
    }
  } else if (!triggerDown) {
    hand.pressedButton = -1;
  }
  hand.triggerDown = triggerDown;

  if (hand.primaryPressed) bridge_.action(h == kRight ? kTogglePause : kCycleAudio);
  if (hand.secondaryPressed) bridge_.action(h == kRight ? kShowProgress : kCycleSubtitles);
  if (hand.menuPressed) bridge_.action(kExit);
  if (hand.stickClickPressed) placeInFront();

  // Thumbstick flicks repeat while held. X seeks 10 s on either hand; Y seeks 5 min on the right
  // hand and changes volume on the left.
  int dirNow = 0;
  const XrVector2f s = hand.stick;
  if (std::fabs(s.x) > 0.7f && std::fabs(s.x) >= std::fabs(s.y)) dirNow = s.x > 0 ? 1 : 2;
  else if (std::fabs(s.y) > 0.7f) dirNow = s.y > 0 ? 3 : 4;
  bool released = std::fabs(s.x) < 0.3f && std::fabs(s.y) < 0.3f;

  auto fire = [&](int d) {
    static const jint rightCodes[] = {0, kSeekForward, kSeekBack, kSeekForwardLong, kSeekBackLong};
    static const jint leftCodes[] = {0, kSeekForward, kSeekBack, kVolumeUp, kVolumeDown};
    bridge_.action(h == kRight ? rightCodes[d] : leftCodes[d]);
  };
  if (hand.stickDir == 0 && dirNow != 0) {
    hand.stickDir = dirNow;
    hand.stickRepeat = 0.45f;
    fire(dirNow);
  } else if (hand.stickDir != 0) {
    if (released) {
      hand.stickDir = 0;
    } else if (dirNow == hand.stickDir) {
      hand.stickRepeat -= dt;
      if (hand.stickRepeat <= 0) {
        hand.stickRepeat = 0.3f;
        fire(dirNow);
      }
    }
  }
}

void XrApp::renderEye(int eyeIndex, const XrView& view, uint32_t imageIndex) {
  EyeTarget& eye = eyes_[eyeIndex];
  GLuint color = eye.images[imageIndex].image;

  glBindFramebuffer(GL_FRAMEBUFFER, eye.fbo);
  if (msaa_) {
    glFramebufferTexture2DMultisampleEXT_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                          color, 0, 4);
  } else {
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
  }
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, eye.depth);

  glViewport(0, 0, eye.width, eye.height);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glDisable(GL_CULL_FACE);
  glDisable(GL_BLEND);
  // Transparent where nothing is drawn so passthrough shows through.
  glClearColor(0, 0, 0, passthroughLayer_ ? 0.0f : 1.0f);
  glClearDepthf(1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  Pose eyePose = fromXr(view.pose);
  Mat4 viewProj = Mat4::projection(view.fov, 0.05f, 100.0f) * Mat4::fromPose(inverse(eyePose));

  ControllerVisual visuals[2];
  for (int h = 0; h < 2; ++h) {
    const Hand& hand = hands_[h];
    // Tracked hands get a laser but no controller model, and no laser while poking.
    bool laser = hand.active || (hand.tracked && hand.aimValid && !hand.nearSet);
    visuals[h].active = laser;
    visuals[h].drawController = hand.active;
    visuals[h].aim = hand.aim;
    if (laser) {
      Vec3 dir = rotate(hand.aim.q, {0, 0, -1});
      float hit = scene_.rayHit(crtPose_, crtScale_, hand.aim.p, dir);
      visuals[h].highlighted = hand.grabbing || (hit >= 0 && hit < 20.0f);
      visuals[h].rayLength = visuals[h].highlighted && hit >= 0 ? hit : 0.6f;
    }
  }

  scene_.draw(viewProj, eyePose.p, crtPose_, crtScale_, visuals, 2);

  if (passthroughLayer_) {
    for (int h = 0; h < 2; ++h) {
      const Hand& hand = hands_[h];
      if (!hand.tracked) continue;
      // Prefer the runtime's skinned hand model; fall back to spheres along the bones.
      bool skinned = scene_.hasHandMesh(h) && hand.bindPoses.size() == CrtScene::kHandJoints;
      Mat4 skin[CrtScene::kHandJoints];
      for (int j = 0; skinned && j < CrtScene::kHandJoints; ++j) {
        const XrHandJointLocationEXT& l = hand.joints[j];
        if (!(l.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) ||
            !(l.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
          skinned = false;
          break;
        }
        skin[j] = Mat4::fromPose(fromXr(l.pose) * inverse(hand.bindPoses[j]));
      }
      if (skinned) {
        scene_.drawHandMesh(h, viewProj, skin, settings_.handPadding);
      } else {
        float spheres[CrtScene::kMaxOccluders * 4];
        int count = collectOccluders(h, spheres, CrtScene::kMaxOccluders);
        scene_.drawOccluders(viewProj, spheres, count);
      }
    }
  }

  const GLenum discard = GL_DEPTH_ATTACHMENT;
  glInvalidateFramebuffer(GL_FRAMEBUFFER, 1, &discard);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void XrApp::frame() {
  XrFrameWaitInfo waitInfo{XR_TYPE_FRAME_WAIT_INFO};
  XrFrameState frameState{XR_TYPE_FRAME_STATE};
  if (XR_FAILED(xrWaitFrame(session_, &waitInfo, &frameState))) return;
  XrFrameBeginInfo beginInfo{XR_TYPE_FRAME_BEGIN_INFO};
  if (XR_FAILED(xrBeginFrame(session_, &beginInfo))) return;

  XrTime now = frameState.predictedDisplayTime;
  float dt = lastTime_ ? std::clamp(static_cast<float>(now - lastTime_) * 1e-9f, 0.0f, 0.1f) : 0.0f;
  lastTime_ = now;

  updateInput(now, dt);

  bool hasFrame = bridge_.updateVideo(texMatrix_);
  scene_.setVideo(videoTexture_, texMatrix_, hasFrame);
  scene_.setPlaying(bridge_.playing());
  uint32_t hovered = 0, pressed = 0;
  for (const Hand& hand : hands_) {
    if (hand.hoverButton >= 0) hovered |= 1u << hand.hoverButton;
    if (hand.pressedButton >= 0) pressed |= 1u << hand.pressedButton;
  }
  scene_.setButtonState(hovered, pressed);

  std::vector<XrCompositionLayerBaseHeader*> layers;
  XrCompositionLayerPassthroughFB passthroughLayer{XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB};
  XrCompositionLayerProjection projection{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
  XrCompositionLayerProjectionView projectionViews[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                                         {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};

  if (passthroughLayer_) {
    passthroughLayer.layerHandle = passthroughLayer_;
    passthroughLayer.flags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&passthroughLayer));
  }

  if (frameState.shouldRender && crtPlaced_) {
    XrViewLocateInfo locateInfo{XR_TYPE_VIEW_LOCATE_INFO};
    locateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    locateInfo.displayTime = now;
    locateInfo.space = appSpace_;
    XrViewState viewState{XR_TYPE_VIEW_STATE};
    XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
    uint32_t viewCount = 0;
    bool located =
        XR_SUCCEEDED(xrLocateViews(session_, &locateInfo, &viewState, 2, &viewCount, views)) &&
        (viewState.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) &&
        (viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT);

    if (located) {
      for (int i = 0; i < 2; ++i) {
        EyeTarget& eye = eyes_[i];
        uint32_t imageIndex = 0;
        XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        xrAcquireSwapchainImage(eye.swapchain, &acquire, &imageIndex);
        XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        wait.timeout = XR_INFINITE_DURATION;
        xrWaitSwapchainImage(eye.swapchain, &wait);

        renderEye(i, views[i], imageIndex);

        XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(eye.swapchain, &release);

        projectionViews[i].pose = views[i].pose;
        projectionViews[i].fov = views[i].fov;
        projectionViews[i].subImage.swapchain = eye.swapchain;
        projectionViews[i].subImage.imageRect = {{0, 0}, {eye.width, eye.height}};
      }
      projection.space = appSpace_;
      projection.viewCount = 2;
      projection.views = projectionViews;
      if (passthroughLayer_) projection.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
      layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&projection));
    }
  }

  XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
  endInfo.displayTime = now;
  endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
  endInfo.layerCount = static_cast<uint32_t>(layers.size());
  endInfo.layers = layers.data();
  xrEndFrame(session_, &endInfo);
}

bool XrApp::run(JNIEnv* env, jobject activity, jobject bridge, jfloatArray initialPose,
                const char* modelPath, jfloatArray settings) {
  settings_.read(env, settings);
  if (!bridge_.init(env, bridge)) return false;

  bool ok = initInstance(env, activity) && initEgl() && initSession() && initSwapchains() &&
            initActions() && scene_.init(modelPath);
  if (!ok) {
    LOGE("XR initialisation failed");
    shutdown();
    bridge_.release();
    return false;
  }
  initPassthrough();
  scene_.setPicture(settings_.scanlines, settings_.scanlineFade, settings_.reflections);
  initHandTracking();

  if (initialPose && env->GetArrayLength(initialPose) == 8 && stageSpace_) {
    float v[8];
    env->GetFloatArrayRegion(initialPose, 0, 8, v);
    crtPose_ = {normalize(Quat{v[3], v[4], v[5], v[6]}), {v[0], v[1], v[2]}};
    crtScale_ = std::clamp(v[7], 0.3f, 3.0f);
    crtPlaced_ = true;
  }

  glGenTextures(1, &videoTexture_);
  glBindTexture(GL_TEXTURE_EXTERNAL_OES, videoTexture_);
  glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_EXTERNAL_OES, 0);
  // Kotlin wraps the texture in a SurfaceTexture and hands its Surface to mpv.
  bridge_.glReady(videoTexture_, scene_.screenAspect());

  bool quit = false;
  while (!quit) {
    pollEvents(quit);
    if (quit) break;

    if (gExitRequested.load() && !exitRequestSent_) {
      exitRequestSent_ = true;
      if (running_) {
        xrRequestExitSession(session_);
      } else {
        break;
      }
    }

    if (!running_) {
      // Idle until the runtime moves us to READY again (or we're told to stop).
      struct timespec ts{0, 10'000'000};
      nanosleep(&ts, nullptr);
      continue;
    }
    frame();
  }

  // Kotlin must detach mpv and release the SurfaceTexture while our context is still current.
  bridge_.sessionEnded();
  shutdown();
  bridge_.release();
  return true;
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_retroarch_browser_retroactivity_XrNative_run(JNIEnv* env, jclass, jobject activity,
                                                         jobject bridge, jfloatArray initialPose,
                                                         jstring modelPath, jfloatArray settings) {
  gExitRequested.store(false);
  std::string path;
  if (modelPath) {
    const char* chars = env->GetStringUTFChars(modelPath, nullptr);
    path = chars;
    env->ReleaseStringUTFChars(modelPath, chars);
  }
  XrApp app;
  return app.run(env, activity, bridge, initialPose, path.c_str(), settings) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_retroarch_browser_retroactivity_XrNative_requestExit(JNIEnv*, jclass) {
  gExitRequested.store(true);
}
