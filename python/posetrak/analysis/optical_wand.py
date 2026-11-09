# SPDX-FileCopyrightText: 2026 Harri Kaimio
# SPDX-License-Identifier: Apache-2.0
"""Optional optical decoder adapter over Posetrak-owned frames and markers."""
from .native_video import VideoFrames
from .shared_pass import SharedAruco


class OpticalWandInput:
    def __init__(self, profile, *, marker_provider=None, native_frames=None, cancelled=lambda: False):
        self.detector = marker_provider or SharedAruco(profile.dictionary)
        self.detector_fingerprint = self.detector.fingerprint
        self.cancelled = cancelled
        self.native_frames = native_frames
        self.stats = {"decoded": 0, "conversions": 0, "passes": 0}

    def __call__(self, video):
        from sync_device.analysis_contract import FrameInput, FrameReference
        source = self.native_frames(video) if self.native_frames else VideoFrames(
            video["path"], video["video_id"], cancelled=self.cancelled, metrics=self.stats)
        for frame in source:
            r = frame.reference
            yield FrameInput(FrameReference(r.video_id, r.ordinal, r.pts, r.time_base,
                                             r.width, r.height, r.pixel_format, r.pts_segment), frame.image)

    def markers(self, frame):
        from sync_device.analysis_contract import MarkerObservation
        return [MarkerObservation(frame.reference, m.dictionary, m.marker_id, m.corners,
                                  m.detector_fingerprint, m.image_transform_fingerprint, m.coordinate_convention)
                for m in self.detector.detect(frame)]


def analyze_optical_wand(request, profile, *, cache=None, progress=lambda event: None,
                         cancelled=lambda: False, marker_provider=None, native_frames=None, workers=1):
    """No session writes; the caller owns result storage and capture selection."""
    from sync_device.wand_runner import analyze_request
    from pathlib import Path
    import hashlib
    import json
    inputs = OpticalWandInput(profile, marker_provider=marker_provider, native_frames=native_frames, cancelled=cancelled)
    marker_cache = None
    if cache and isinstance(inputs.detector, SharedAruco):
        sources = {}
        for video in request["videos"]:
            if cancelled():
                raise InterruptedError("Optical analysis cancelled")
            with Path(video["path"]).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            sources[video["video_id"]] = {"sha256": digest, "mode_id": video["mode_id"],
                                         "intrinsics_id": video["intrinsics_id"], "mode": video["mode"]}
        key = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        marker_cache = Path(cache) / f"markers-{key}-{inputs.detector.fingerprint}.json"
        if marker_cache.exists():
            inputs.detector.load(marker_cache, sources)
    result = analyze_request(request, profile, inputs, inputs.markers, cache=cache,
                             progress=progress, cancelled=cancelled,
                             detector_fingerprint=inputs.detector_fingerprint, workers=workers)
    if marker_cache:
        inputs.detector.save(marker_cache, sources)
    metrics = dict(inputs.stats, marker_detections=getattr(inputs.detector, "calls", None))
    return result, metrics


def main():
    """Run offline analysis without importing the capture daemon or Bluetooth."""
    import argparse
    import json
    from pathlib import Path
    import cv2
    from sync_device.wand_profile import WandProfile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cv2.setNumThreads(2)
    result, metrics = analyze_optical_wand(json.loads(args.request.read_text(encoding="utf-8")),
        WandProfile.load(args.profile), cache=args.cache, workers=args.workers,
        progress=lambda e: print(json.dumps(e), flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".performance.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
