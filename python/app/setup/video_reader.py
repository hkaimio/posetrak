"""video_reader.py — Background video frame decoder for Qt UI widgets."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QThread, Signal


class FrameReader(QThread):
    """Decodes individual video frames in a background thread.

    Rapid requests coalesce: only the most recently requested frame index
    is decoded.  Callers submit work via ``request()`` and receive results
    via the ``frame_ready`` signal.

    Usage::

        reader = FrameReader(file_path, parent=widget)
        reader.frame_ready.connect(on_frame)
        reader.start()
        reader.request(42)
        # … later …
        reader.shutdown()
    """

    frame_ready = Signal(int, object)  # (frame_idx, numpy_array)

    def __init__(self, file_path: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._file_path = file_path
        self._pending: int | None = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = False

    def request(self, frame_idx: int) -> None:
        with self._lock:
            self._pending = frame_idx
        self._event.set()

    def shutdown(self) -> None:
        self._stop = True
        self._event.set()
        self.wait(2000)

    def run(self) -> None:
        import cv2
        cap = cv2.VideoCapture(str(self._file_path))
        while not self._stop:
            self._event.wait()
            self._event.clear()
            if self._stop:
                break
            with self._lock:
                idx = self._pending
            if idx is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(idx, frame)
        cap.release()
