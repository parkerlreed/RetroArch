package com.retroarch.browser.retroactivity;

import android.app.Activity;

/** The OpenXR renderer in libretroarch-xr. */
final class XrNative
{
  static
  {
    System.loadLibrary("retroarch-xr");
  }

  private XrNative() {}

  /**
   * Runs the headset session until it ends, on the calling thread, which becomes the
   * render thread. The bridge gets onGlReady, updateVideoTexture, isPlaying, onAction,
   * onCrtPoseChanged and onSessionEnded calls on that thread.
   * Returns false if OpenXR is unavailable.
   */
  static native boolean run(Activity activity, Object bridge, float[] initialPose,
      String modelPath, float[] settings);

  static native void requestExit();
}
