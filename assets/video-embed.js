/* SPDX-FileCopyrightText: 2026 Harri Kaimio
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Scroll-triggered play/pause for embedded Vimeo videos (see
 * video-embed.css) via the Vimeo Player SDK + IntersectionObserver: a
 * video starts playing once it's mostly scrolled into view and pauses
 * again once it scrolls back out, instead of autoplaying immediately on
 * page load (annoying with several videos on one page) or requiring the
 * visitor to click play themselves. Embeds should include muted=1&loop=1
 * in their iframe src -- muted because browsers block audible autoplay
 * triggered by anything other than a direct user gesture (scrolling
 * doesn't count), loop so a video that finishes replays instead of
 * showing Vimeo's end-of-video "more from this creator" overlay.
 *
 * Registered as extra_javascript in mkdocs.yml, after the Vimeo Player
 * SDK. Wires up via document$ rather than a plain DOMContentLoaded
 * listener because this theme has navigation.instant enabled: Material's
 * client-side page swaps don't fire a fresh DOMContentLoaded, so a plain
 * listener would only ever wire up videos on whichever page happened to
 * be loaded first.
 */
function initVideoEmbeds() {
  if (typeof Vimeo === "undefined") return;

  var wrappers = document.querySelectorAll(".video-embed");
  if (!wrappers.length) return;

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        var player = entry.target._vimeoPlayer;
        if (!player) return;
        var action = entry.isIntersecting ? "play" : "pause";
        player[action]().catch(function () {
          /* autoplay blocked, player already in that state, etc. -- not
             worth surfacing to the visitor */
        });
      });
    },
    { threshold: 0.5 }
  );

  wrappers.forEach(function (wrapper) {
    if (wrapper._vimeoPlayer) return; // already wired up
    var iframe = wrapper.querySelector("iframe");
    if (!iframe) return;
    wrapper._vimeoPlayer = new Vimeo.Player(iframe);
    observer.observe(wrapper);
  });
}

if (window.document$) {
  document$.subscribe(initVideoEmbeds);
} else {
  document.addEventListener("DOMContentLoaded", initVideoEmbeds);
}
