# SPDX-FileCopyrightText: 2026 Harri Kaimio
# SPDX-License-Identifier: Apache-2.0
"""Synchronous bounded fan-out with reusable native-domain ArUco observations."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import cv2
from threading import RLock
from fractions import Fraction
from pathlib import Path
from .native_video import FrameIdentity

from app.setup.fiducial_markers import ArucoDetector


@dataclass(frozen=True)
class MarkerCorners:
    reference: object
    dictionary: str
    marker_id: int
    corners: tuple
    detector_fingerprint: str
    image_transform_fingerprint: str = "native-v1"
    coordinate_convention: str = "native-distorted-pixels"


class SharedAruco:
    def __init__(self, dictionary, *, cache_size=4096):
        self.dictionary = dictionary
        self.detector = ArucoDetector(dictionary)
        value = {"opencv": cv2.__version__, "dictionary": dictionary, "settings": "opencv-default-v1"}
        self.fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.cache_size, self.cache, self.calls = cache_size, OrderedDict(), 0
        self.lock = RLock()

    def detect(self, frame):
        with self.lock:
            return self._detect(frame)

    def _detect(self, frame):
        ref = frame.reference
        key = (ref.video_id, ref.ordinal, ref.pts, str(ref.time_base), ref.width, ref.height, ref.pixel_format, ref.pts_segment)
        if key not in self.cache:
            detections = self.detector.detect(frame.image.bgr(), ref.video_id, ref.ordinal)
            self.calls += 1
            counts = {d.marker_id: sum(o.marker_id == d.marker_id for o in detections) for d in detections}
            self.cache[key] = tuple(MarkerCorners(ref, self.dictionary, int(d.marker_id),
                tuple((float(c.px), float(c.py)) for c in sorted(d.corners, key=lambda c: c.corner_index)), self.fingerprint)
                for d in detections if counts[d.marker_id] == 1)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def save(self, path, sources):
        """Persist native marker observations with source/mode provenance."""
        with self.lock:
            records = [{"frame": list(key), "markers": [{"id": m.marker_id, "corners": m.corners} for m in markers]}
                       for key, markers in self.cache.items()]
        value = {"schema": "posetrak.native-aruco-observations.v1", "detector_fingerprint": self.fingerprint,
                 "sources": sources, "records": records}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def load(self, path, sources):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if (value.get("schema") != "posetrak.native-aruco-observations.v1"
                or value.get("sources") != sources or value.get("detector_fingerprint") != self.fingerprint):
            raise ValueError("Incompatible marker observation cache provenance")
        loaded = OrderedDict()
        for row in value["records"][-self.cache_size:]:
            key = tuple(row["frame"])
            if key[0] not in sources:
                raise ValueError("Marker cache contains an unknown source")
            ref = FrameIdentity(key[0], key[1], key[2], Fraction(key[3]), key[4], key[5], key[6], key[7])
            loaded[key] = tuple(MarkerCorners(ref, self.dictionary, m["id"], tuple(tuple(p) for p in m["corners"]), self.fingerprint)
                                for m in row["markers"])
        with self.lock:
            self.cache = loaded


@dataclass(frozen=True)
class Consumer:
    callback: object
    stride: int = 1
    markers: bool = False


def run_shared_pass(frames, consumers, detector, *, cancelled=lambda: False, progress=lambda ref: None):
    """Decode once, detect at most once per frame, invoke consumers synchronously.

    No image queue is retained. Callbacks must copy any image data they keep.
    Independent local-frame analyses do not need a synchronization result.
    """
    if any(c.stride < 1 for c in consumers):
        raise ValueError("Consumer stride must be positive")
    for frame in frames:
        if cancelled():
            raise InterruptedError("Shared analysis cancelled")
        active = [c for c in consumers if frame.reference.ordinal % c.stride == 0]
        observations = detector.detect(frame) if any(c.markers for c in active) else ()
        for consumer in active:
            consumer.callback(frame, observations if consumer.markers else ())
        progress(frame.reference)
