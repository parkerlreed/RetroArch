package com.retroarch.browser.retroactivity;

import android.app.Activity;
import android.content.SharedPreferences;
import android.graphics.SurfaceTexture;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Surface;

import java.io.File;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Shows RetroArch on a CRT placed in the room, in passthrough. RetroArch renders into a
 * SurfaceTexture owned by the headset renderer instead of the activity window.
 */
public final class XrSession
{
  private static final String TAG = "RetroArch-XR";
  private static final String MODEL_FILE_NAME = "tv.glb";
  private static final String POSE_PREFS = "xr";
  private static final String KEY_TV_POSE = "tv_pose";
  private static final int VIDEO_HEIGHT = 1080;
  private static final long JOIN_TIMEOUT_MS = 3000;
  /* Action codes from xr_actions.h. */
  private static final int ACTION_POWER = 9;

  /* Hand padding (m), room opacity, room saturation, scanlines, scanline fade,
   * glass reflections, menu-gesture distance (m). Order matches xr_app.cpp.
   * No scanlines: RetroArch's own shaders handle the picture. */
  private static final float[] SETTINGS = { 0.004f, 1f, 1f, 0f, 0f, 1f, 0.6f };

  private static XrSession current;

  private final Activity activity;
  private final AtomicBoolean frameAvailable = new AtomicBoolean(false);
  private Thread renderThread;
  private Thread attachThread;

  /* Render thread only. */
  private SurfaceTexture surfaceTexture;
  private Surface surface;
  private boolean hasFrame;

  private XrSession(Activity activity)
  {
    this.activity = activity;
  }

  public static void onCreate(Activity activity)
  {
    if (!isQuest() || current != null)
      return;
    /* Before the activity window arrives, so RetroArch never binds to it. */
    nativeSetXrEnabled(true);
    current = new XrSession(activity);
    current.start();
  }

  public static void onDestroy(Activity activity)
  {
    XrSession session = current;
    if (session == null || session.activity != activity)
      return;
    XrNative.requestExit();
    join(session.renderThread);
    current = null;
  }

  private static boolean isQuest()
  {
    return "Oculus".equalsIgnoreCase(Build.MANUFACTURER)
        || "Meta".equalsIgnoreCase(Build.MANUFACTURER);
  }

  private static void join(Thread thread)
  {
    if (thread == null)
      return;
    try
    {
      thread.join(JOIN_TIMEOUT_MS);
    }
    catch (InterruptedException e)
    {
      Thread.currentThread().interrupt();
    }
  }

  private void start()
  {
    final float[] pose = loadPose();
    final File model = new File(activity.getExternalFilesDir(null), MODEL_FILE_NAME);
    final String modelPath = model.isFile() ? model.getPath() : null;
    renderThread = new Thread(new Runnable()
    {
      @Override
      public void run()
      {
        if (!XrNative.run(activity, XrSession.this, pose, modelPath, SETTINGS))
        {
          Log.w(TAG, "OpenXR unavailable, using the normal window");
          activity.runOnUiThread(new Runnable()
          {
            @Override
            public void run()
            {
              nativeSetXrEnabled(false);
            }
          });
        }
      }
    }, "RetroArch-XR");
    renderThread.start();
  }

  /* ---- Called by the renderer, on its thread. ---- */

  void onGlReady(int texture, float screenAspect)
  {
    int width = Math.max(VIDEO_HEIGHT, Math.min(VIDEO_HEIGHT * 2,
        (int) (VIDEO_HEIGHT * screenAspect) / 2 * 2));
    surfaceTexture = new SurfaceTexture(texture);
    surfaceTexture.setDefaultBufferSize(width, VIDEO_HEIGHT);
    surfaceTexture.setOnFrameAvailableListener(new SurfaceTexture.OnFrameAvailableListener()
    {
      @Override
      public void onFrameAvailable(SurfaceTexture st)
      {
        frameAvailable.set(true);
      }
    }, new Handler(Looper.getMainLooper()));
    surface = new Surface(surfaceTexture);

    /* Handing over the window waits for RetroArch's thread to take it, so keep that off
     * the render loop. */
    final Surface target = surface;
    attachThread = new Thread(new Runnable()
    {
      @Override
      public void run()
      {
        nativeSetXrSurface(target);
      }
    }, "RetroArch-XR-attach");
    attachThread.start();
  }

  boolean updateVideoTexture(float[] matrix)
  {
    if (surfaceTexture == null)
      return false;
    if (frameAvailable.getAndSet(false))
    {
      surfaceTexture.updateTexImage();
      hasFrame = true;
    }
    surfaceTexture.getTransformMatrix(matrix);
    return hasFrame;
  }

  boolean isPlaying()
  {
    return true;
  }

  void onAction(int code)
  {
    if (code == ACTION_POWER)
      nativeToggleMenu();
  }

  void onCrtPoseChanged(float[] pose)
  {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < pose.length; i++)
    {
      if (i > 0)
        sb.append(',');
      sb.append(pose[i]);
    }
    prefs().edit().putString(KEY_TV_POSE, sb.toString()).apply();
  }

  void onSessionEnded()
  {
    join(attachThread);
    /* RetroArch must let go of the surface before the texture goes away. */
    if (surface != null)
      nativeSetXrSurface(null);
    if (surface != null)
      surface.release();
    if (surfaceTexture != null)
      surfaceTexture.release();
    surface = null;
    surfaceTexture = null;
    activity.runOnUiThread(new Runnable()
    {
      @Override
      public void run()
      {
        if (!activity.isFinishing())
          activity.finish();
      }
    });
  }

  private SharedPreferences prefs()
  {
    return activity.getSharedPreferences(POSE_PREFS, Activity.MODE_PRIVATE);
  }

  private float[] loadPose()
  {
    String saved = prefs().getString(KEY_TV_POSE, null);
    if (saved == null)
      return null;
    String[] parts = saved.split(",");
    if (parts.length != 8)
      return null;
    float[] pose = new float[8];
    try
    {
      for (int i = 0; i < 8; i++)
        pose[i] = Float.parseFloat(parts[i]);
    }
    catch (NumberFormatException e)
    {
      return null;
    }
    return pose;
  }

  private static native void nativeSetXrEnabled(boolean enabled);

  private static native void nativeSetXrSurface(Surface surface);

  private static native void nativeToggleMenu();
}
