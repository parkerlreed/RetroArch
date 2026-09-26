package com.retroarch.browser.retroactivity;

import android.app.Activity;

/** No-op outside the Quest build. */
public final class XrSession
{
  private XrSession() {}

  public static void onCreate(Activity activity) {}

  public static void onDestroy(Activity activity) {}
}
