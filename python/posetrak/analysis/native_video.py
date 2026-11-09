# SPDX-FileCopyrightText: 2026 Harri Kaimio
# SPDX-License-Identifier: Apache-2.0
"""Sequential native PTS identities and lazy read-only BGR/crop conversion."""
from dataclasses import dataclass
from fractions import Fraction
import av


@dataclass(frozen=True)
class FrameIdentity:
    video_id: str
    ordinal: int
    pts: int
    time_base: Fraction
    width: int
    height: int
    pixel_format: str = "bgr24"
    pts_segment: int = 0


class NativeImage:
    """A borrowed image, valid for synchronous consumers before stream advance."""

    def __init__(self, frame, graphs, metrics):
        self._frame, self._graphs, self._metrics = frame, graphs, metrics
        self._converted = {}

    def bgr(self, region=None):
        if region not in self._converted:
            if region is None:
                image = self._frame.to_ndarray(format="bgr24")
            else:
                x, y, w, h = region
                if min(x, y) < 0 or min(w, h) <= 0 or x+w > self._frame.width or y+h > self._frame.height:
                    raise ValueError("Crop outside native image")
                if any(v % 2 for v in region):
                    # Subsampled YUV crop filters round odd origins; slice native BGR instead.
                    image = self.bgr()[y:y+h, x:x+w]
                    self._converted[region] = image
                    return image
                if region not in self._graphs:
                    if len(self._graphs) >= 8:
                        self._graphs.pop(next(iter(self._graphs)))
                    graph = av.filter.Graph()
                    source = graph.add_buffer(template=self._frame)
                    crop = graph.add("crop", f"{w}:{h}:{x}:{y}")
                    sink = graph.add("buffersink")
                    source.link_to(crop)
                    crop.link_to(sink)
                    graph.configure()
                    self._graphs[region] = graph, source, sink
                _, source, sink = self._graphs[region]
                source.push(self._frame)
                image = sink.pull().to_ndarray(format="bgr24")
            image.flags.writeable = False
            self._converted[region] = image
            self._metrics["conversions"] += 1
        return self._converted[region]


@dataclass(frozen=True)
class NativeFrame:
    reference: FrameIdentity
    image: NativeImage
    discontinuity_before: bool = False


class VideoFrames:
    """Own decoding; fan-out consumers must not retain borrowed frame images.

    Ordinals are counted in decoded presentation order, never reconstructed from
    nominal FPS. A native PTS reset starts a separate index segment.
    """

    def __init__(self, path, video_id, *, cancelled=lambda: False, metrics=None):
        self.path, self.video_id, self.cancelled = str(path), video_id, cancelled
        self.metrics = metrics if metrics is not None else {"decoded": 0, "conversions": 0, "passes": 0}
        self.index = []
        self.complete = False

    def __iter__(self):
        self.index, self.complete = [], False
        self.metrics["passes"] += 1
        with av.open(self.path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "SLICE"
            stream.codec_context.thread_count = 4
            previous, shape, segment, graphs = None, None, 0, {}
            for ordinal, frame in enumerate(container.decode(stream)):
                if self.cancelled():
                    raise InterruptedError("Video analysis cancelled")
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Video frame has no native PTS/time base")
                if int(stream.metadata.get("rotate", "0")) % 360:
                    raise ValueError("Rotation requires an explicit analysis image transform")
                # Preserve raw native pixels; do not silently apply display matrices.
                for data in frame.side_data:
                    if "DISPLAYMATRIX" in str(data.type).upper():
                        from posetrak.detection.frame_source import _parse_displaymatrix
                        if _parse_displaymatrix(bytes(data)):
                            raise ValueError("Display rotation requires an explicit analysis image transform")
                current = frame.pts * frame.time_base
                changed = previous is not None and (current <= previous or shape != (frame.width, frame.height))
                if changed:
                    segment += 1
                    graphs = {}
                ref = FrameIdentity(self.video_id, ordinal, frame.pts, Fraction(frame.time_base),
                                    frame.width, frame.height, pts_segment=segment)
                self.index.append(ref)
                self.metrics["decoded"] += 1
                previous, shape = current, (frame.width, frame.height)
                yield NativeFrame(ref, NativeImage(frame, graphs, self.metrics), changed)
            self.complete = True
