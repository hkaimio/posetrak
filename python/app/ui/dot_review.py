# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Dot-assignment review: check and correct which marker slot each dot tracklet is.

    python -m app.ui.dot_review SESSION_DB TRIAL DETECTION_RUN ATTACHMENT.yaml

A dope sheet shows, for one camera at a time, a row per marker slot (grouped by body
part, groups collapsible) and a bar per tracklet on its slot's row. A bar is coloured by
how far its automatic proposal can be trusted and gets a white border once a person has
confirmed or changed it. The frame view shows the tracklets of the camera around the
selected one, with their slots; clicking a ring selects that tracklet. Right-click a bar
or ring to assign it to a slot, forbid its slot, mark it as not a marker, or clear the
edits made to it. Edits are stored per trial (posetrak.markers.review).

Keys: N next tracklet to review, Z whole frame / zoom to the selection, A accept the proposal, Delete reject, Ctrl+Z undo,
Left/Right step a frame.
"""
from __future__ import annotations

import bisect
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from app.pose.frame_service import FrameService
from posetrak.calibration.session_cameras import load_sync_table
from posetrak.markers.review import (
    Edit, Proposal, add_edit, effective_slot, load_edits, load_proposals, remove_edit, segments,
)
from posetrak.markers.tracklets import collect_tracklets

ROW_H = 18
LABEL_W = 170
RULER_H = 22
UNASSIGNED = "Unassigned"
PROPOSAL_COLOURS = {"high": "#46a05a", "medium": "#c8aa32", "review": "#d25a3c", "none": "#787878"}
EDIT_COLOUR = "#2f6fe0"


@dataclass
class Tracklet:
    camera_id: str
    tracklet_id: int
    first_frame: int
    last_frame: int
    frames: dict            # frame -> (x, y) distorted pixels
    proposal: Proposal | None


def mirror_group(name: str) -> str:
    """The same body part on the other side: forearm.L -> forearm.R."""
    for a, b in ((".L", ".R"), (".R", ".L"), ("_L", "_R"), ("_R", "_L")):
        if name.endswith(a):
            return name[: -len(a)] + b
    return name


class ReviewData:
    """Proposals, edits and tracklet positions of one trial, and the edits' persistence."""

    def __init__(self, conn, trial_id: str, detection_run_id: str, attachment: str) -> None:
        self.conn, self.trial_id, self.detection_run_id = conn, trial_id, detection_run_id
        shot_id = conn.execute("SELECT capture_id FROM trials WHERE id=?", (trial_id,)).fetchone()[0]
        self.sync, self.svid_by_cam = load_sync_table(conn, shot_id)
        self.labels = {r["id"]: r["label"] for r in conn.execute("SELECT id, label FROM camera_instances")}
        self.videos = {r["camera_instance_id"]: r["file_path"]
                       for r in conn.execute("SELECT camera_instance_id, file_path FROM capture_videos WHERE shot_id=?", (shot_id,))}
        markers = yaml.safe_load(open(attachment, encoding="utf-8"))["markers"]
        self.groups: dict[str, list[str]] = {}
        for m in markers:
            self.groups.setdefault(m["parent_joint"], []).append(m["name"])
        self.group_of = {s: g for g, slots in self.groups.items() for s in slots}
        proposals = {(p.camera_id, p.tracklet_id): p for p in load_proposals(conn, detection_run_id)}
        self.tracklets: dict[str, dict[int, Tracklet]] = {}
        for cam in sorted({c for c, _ in proposals}):
            mine = {tid: p for (c, tid), p in proposals.items() if c == cam}
            lo, hi = min(p.first_frame for p in mine.values()), max(p.last_frame for p in mine.values())
            found = collect_tracklets(conn, detection_run_id, self.svid_by_cam[cam], lo, hi)
            self.tracklets[cam] = {tid: Tracklet(cam, tid, p.first_frame, p.last_frame, found.get(tid, {}), p)
                                   for tid, p in mine.items()}
        self.edits: dict[tuple[str, int], list[Edit]] = {}
        for e in load_edits(conn, trial_id, detection_run_id):
            self.edits.setdefault((e.camera_id, e.tracklet_id), []).append(e)
        self.undo: list[tuple[str, str, Edit | None]] = []     # ("add"|"remove", edit_id, edit)

    def time_of(self, cam: str, frame: int) -> float:
        return self.sync.frame_to_global_time(frame, self.svid_by_cam[cam])

    def frame_at(self, cam: str, t: float) -> int:
        return int(self.sync.lookup(t, self.svid_by_cam[cam]))

    def edits_of(self, t: Tracklet) -> list[Edit]:
        return self.edits.get((t.camera_id, t.tracklet_id), [])

    def segments_of(self, t: Tracklet):
        return segments(t.first_frame, t.last_frame, t.proposal, self.edits_of(t))

    def slot_at(self, t: Tracklet, frame: int) -> tuple[str | None, str]:
        return effective_slot(t.proposal, self.edits_of(t), frame)

    def needs_review(self, t: Tracklet) -> bool:
        return not self.edits_of(t) and t.proposal.confidence != "high" and t.proposal.slot is not None

    def add(self, t: Tracklet, kind: str, slot: str | None, first: int | None = None, last: int | None = None) -> None:
        edit = Edit(t.camera_id, t.tracklet_id, kind, slot, first, last)
        edit_id = add_edit(self.conn, self.trial_id, self.detection_run_id, edit)
        stored = Edit(**{**edit.__dict__, "id": edit_id})
        self.edits.setdefault((t.camera_id, t.tracklet_id), []).append(stored)
        self.undo.append(("add", edit_id, stored))

    def clear(self, t: Tracklet) -> None:
        for e in list(self.edits_of(t)):
            remove_edit(self.conn, e.id)
            self.undo.append(("remove", e.id, e))
        self.edits.pop((t.camera_id, t.tracklet_id), None)

    def undo_last(self) -> bool:
        if not self.undo:
            return False
        action, edit_id, edit = self.undo.pop()
        key = (edit.camera_id, edit.tracklet_id)
        if action == "add":
            remove_edit(self.conn, edit_id)
            self.edits[key] = [e for e in self.edits.get(key, []) if e.id != edit_id]
        else:                       # a cleared edit comes back, as a new row
            new_id = add_edit(self.conn, self.trial_id, self.detection_run_id, edit)
            self.edits.setdefault(key, []).append(Edit(**{**edit.__dict__, "id": new_id}))
        return True


class DopeSheet(QWidget):
    """Slots on rows, time to the right; one camera's tracklets as bars."""

    selected = Signal(object)          # Tracklet or None
    time_picked = Signal(float)
    context_requested = Signal(object, object)   # Tracklet, global QPoint

    def __init__(self, data: ReviewData) -> None:
        super().__init__()
        self.data, self.camera = data, None
        self.collapsed: set[str] = set()
        self.show_none = True
        self._lane_of: dict[tuple[int, int], int] = {}     # (tracklet id, first frame of segment) -> lane
        self.playhead = 0.0
        self.t_left, self.px_per_s, self.y_off = 0.0, 40.0, 0
        self.selection: Tracklet | None = None
        self._hits: list[tuple[QRectF, Tracklet]] = []
        self._rows: list[tuple[str, str]] = []
        self.setMinimumSize(500, 300)
        self.setFocusPolicy(Qt.StrongFocus)
        self.set_camera(next(iter(data.tracklets)))

    def set_camera(self, cam: str) -> None:
        self.camera = cam
        starts = [self.data.time_of(cam, t.first_frame) for t in self.data.tracklets[cam].values()]
        self.t_left = min(starts) - 0.5 if starts else 0.0
        self._build_rows()
        self.update()

    def _build_rows(self) -> None:
        self._rows = []
        for group, slots in self.data.groups.items():
            self._rows.append(("group", group))
            if group not in self.collapsed:
                self._rows += [("slot", s) for s in slots]
        lanes = self._pack_unassigned()
        if self.show_none:
            self._rows.append(("group", UNASSIGNED))
            if UNASSIGNED not in self.collapsed:
                self._rows += [("lane", str(i)) for i in range(lanes)]

    def _pack_unassigned(self) -> int:
        """Stack the segments without a slot into as few lanes as hold them without overlap."""
        segs = sorted((self.data.time_of(self.camera, f0), self.data.time_of(self.camera, f1 + 1), t.tracklet_id, f0)
                      for t in self.data.tracklets[self.camera].values()
                      for f0, f1, slot, _ in self.data.segments_of(t) if slot is None)
        ends: list[float] = []
        self._lane_of = {}
        for start, end, tid, f0 in segs:
            lane = next((i for i, e in enumerate(ends) if e <= start), None)
            if lane is None:
                ends.append(end); lane = len(ends) - 1
            else:
                ends[lane] = end
            self._lane_of[(tid, f0)] = lane
        return len(ends)

    def refresh(self) -> None:
        """Rebuild the rows after edits changed which tracklets lack a slot."""
        self._build_rows()
        self.update()

    def _row_of(self, slot: str | None, t: Tracklet | None = None, f0: int = 0) -> int | None:
        if slot is None:
            lane = self._lane_of.get((t.tracklet_id, f0)) if t is not None else None
            key = ("lane", str(lane))
        else:
            key = ("slot", slot)
        for i, row in enumerate(self._rows):
            if row == key:
                return i
        return None

    def x_of(self, t: float) -> float:
        return LABEL_W + (t - self.t_left) * self.px_per_s

    def t_of(self, x: float) -> float:
        return self.t_left + (x - LABEL_W) / self.px_per_s

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1e1f22"))
        font = QFont(self.font()); font.setPointSize(8); p.setFont(font)
        self._hits = []
        top = RULER_H
        for i, (kind, name) in enumerate(self._rows):
            y = top + i * ROW_H - self.y_off
            if y + ROW_H < top or y > self.height():
                continue
            if kind == "lane":
                p.setPen(QColor("#2f3136")); p.drawLine(LABEL_W, y + ROW_H, self.width(), y + ROW_H)
            elif kind == "group":
                p.fillRect(0, y, self.width(), ROW_H, QColor("#2b2d31"))
                p.setPen(QColor("#d0d0d0"))
                p.drawText(6, y, LABEL_W, ROW_H, Qt.AlignVCenter, ("▸ " if name in self.collapsed else "▾ ") + name)
            else:
                p.setPen(QColor("#2f3136")); p.drawLine(LABEL_W, y + ROW_H, self.width(), y + ROW_H)
                p.setPen(QColor("#a0a0a0")); p.drawText(18, y, LABEL_W - 18, ROW_H, Qt.AlignVCenter, name)
        for t in self.data.tracklets[self.camera].values():
            for f0, f1, slot, origin in self.data.segments_of(t):
                row = self._row_of(slot, t, f0)
                if row is None:
                    continue
                y = top + row * ROW_H - self.y_off
                x0, x1 = self.x_of(self.data.time_of(self.camera, f0)), self.x_of(self.data.time_of(self.camera, f1 + 1))
                if x1 < LABEL_W or x0 > self.width() or y + ROW_H < top or y > self.height():
                    continue
                rect = QRectF(max(x0, LABEL_W), y + 2, max(x1 - max(x0, LABEL_W), 3.0), ROW_H - 4)
                colour = EDIT_COLOUR if origin == "edit" else PROPOSAL_COLOURS[t.proposal.confidence]
                p.setBrush(QBrush(QColor(colour)))
                if t is self.selection:
                    p.setPen(QPen(QColor("#ffe040"), 2))
                elif self.data.edits_of(t):
                    p.setPen(QPen(QColor("#ffffff"), 2))
                else:
                    p.setPen(QPen(QColor("#101010"), 1))
                p.drawRect(rect)
                if rect.width() > 36:
                    p.setPen(QColor("#ffffff")); p.drawText(rect, Qt.AlignCenter, str(t.tracklet_id))
                self._hits.append((rect, t))
        p.fillRect(0, 0, self.width(), RULER_H, QColor("#2b2d31"))
        p.setPen(QColor("#c0c0c0"))
        step = 1 if self.px_per_s >= 30 else 5 if self.px_per_s >= 6 else 20
        for sec in range(int(self.t_left // step) * step, int(self.t_of(self.width())) + 1, step):
            x = self.x_of(sec)
            if x >= LABEL_W:
                p.drawLine(int(x), RULER_H - 6, int(x), RULER_H)
                p.drawText(int(x) + 3, 14, f"{sec}")
        x = self.x_of(self.playhead)
        if x >= LABEL_W:
            p.setPen(QPen(QColor("#ff5050"), 1)); p.drawLine(int(x), 0, int(x), self.height())

    def _tracklet_at(self, pos) -> Tracklet | None:
        for rect, t in reversed(self._hits):
            if rect.contains(QPointF(pos)):
                return t
        return None

    def mousePressEvent(self, e) -> None:
        pos = e.position().toPoint()
        t = self._tracklet_at(pos)
        if e.button() == Qt.RightButton:
            if t is not None:
                self.select(t)
                self.context_requested.emit(t, e.globalPosition().toPoint())
            return
        if pos.x() < LABEL_W:
            row = (pos.y() - RULER_H + self.y_off) // ROW_H
            if 0 <= row < len(self._rows) and self._rows[row][0] == "group":
                name = self._rows[row][1]
                self.collapsed ^= {name}
                self._build_rows(); self.update()
            return
        if t is not None:
            self.select(t)
        if pos.x() >= LABEL_W:
            self.time_picked.emit(max(self.t_of(pos.x()), 0.0))

    def select(self, t: Tracklet | None) -> None:
        self.selection = t
        self.selected.emit(t)
        self.update()

    def wheelEvent(self, e) -> None:
        dy = e.angleDelta().y()
        if e.modifiers() & Qt.ControlModifier:
            anchor = self.t_of(e.position().x())
            self.px_per_s = min(max(self.px_per_s * (1.2 if dy > 0 else 1 / 1.2), 2.0), 400.0)
            self.t_left = anchor - (e.position().x() - LABEL_W) / self.px_per_s
        elif e.modifiers() & Qt.ShiftModifier:
            self.t_left -= dy / self.px_per_s
        else:
            limit = max(0, len(self._rows) * ROW_H - (self.height() - RULER_H))
            self.y_off = min(max(self.y_off - dy // 2, 0), limit)
        self.update()

    def reveal(self, t: Tracklet) -> None:
        """Scroll so the tracklet's first segment is in view."""
        slot = self.data.slot_at(t, t.first_frame)[0]
        group = self.data.group_of.get(slot, UNASSIGNED)
        if group in self.collapsed:
            self.collapsed.discard(group); self._build_rows()
        row = self._row_of(slot, t, t.first_frame)
        if row is not None:
            self.y_off = max(0, row * ROW_H - (self.height() - RULER_H) // 2)
        start = self.data.time_of(t.camera_id, t.first_frame)
        if not (self.t_of(LABEL_W) <= start <= self.t_of(self.width() - 100)):
            self.t_left = start - 2.0


class FramePane(QWidget):
    """The camera's frame around the selected tracklet, with the slot of every tracklet near it."""

    tracklet_picked = Signal(object)
    context_requested = Signal(object, object)
    CROP = 360                   # source pixels shown on the longer side of the pane

    def __init__(self, data: ReviewData, service: FrameService) -> None:
        super().__init__()
        self.data, self.service = data, service
        self.camera, self.t, self.selection = None, 0.0, None
        self.zoomed = True               # around the selected tracklet, or the whole frame
        self._rings: list[tuple[QPointF, Tracklet]] = []
        self.setMinimumSize(420, 420)

    def show_at(self, camera: str, t: float, selection: Tracklet | None) -> None:
        self.camera, self.t, self.selection = camera, t, selection
        self.service.request(self.data.svid_by_cam[camera], self.data.frame_at(camera, t))
        self.update()

    def _nearest_position(self, tr: Tracklet, frame: int):
        frames = tr.frames
        if frame in frames:
            return frames[frame]
        order = sorted(frames)
        k = bisect.bisect_left(order, frame)
        near = [order[i] for i in (k - 1, k) if 0 <= i < len(order) and abs(order[i] - frame) <= 3]
        return frames[min(near, key=lambda f: abs(f - frame))] if near else None

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#101010"))
        self._rings = []
        if self.camera is None:
            return
        svid = self.data.svid_by_cam[self.camera]
        frame = self.data.frame_at(self.camera, self.t)
        got = self.service.get(svid, frame)
        focus = self._nearest_position(self.selection, frame) if self.selection and self.zoomed else None
        if focus is not None and got is not None:
            r = self.CROP
            self.service.request_region(svid, frame, (focus[0] - r / 2, focus[1] - r / 2, focus[0] + r / 2, focus[1] + r / 2), 1.0)
            got = self.service.get_region(svid, frame) or got
        if got is None:
            p.setPen(QColor("#a0a0a0")); p.drawText(self.rect(), Qt.AlignCenter, "decoding…")
            return
        jpeg, w, h, sx, sy, sw, sh = got
        image = QImage.fromData(jpeg)
        scale_img = w / sw                                  # image pixels per source pixel
        cx, cy = focus if focus else (sx + sw / 2, sy + sh / 2)
        side = min(self.width(), self.height())
        view = self.CROP if focus else max(sw, sh)
        zoom = side / view                                   # widget pixels per source pixel
        src = QRectF((cx - sx) * scale_img - view * scale_img / 2, (cy - sy) * scale_img - view * scale_img / 2,
                     view * scale_img, view * scale_img)
        p.drawImage(QRectF(0, 0, side, side), image, src)
        for tr in self.data.tracklets[self.camera].values():
            xy = self._nearest_position(tr, frame) if tr.first_frame - 3 <= frame <= tr.last_frame + 3 else None
            if xy is None:
                continue
            pt = QPointF((xy[0] - cx) * zoom + side / 2, (xy[1] - cy) * zoom + side / 2)
            if not (0 <= pt.x() <= side and 0 <= pt.y() <= side):
                continue
            slot, origin = self.data.slot_at(tr, min(max(frame, tr.first_frame), tr.last_frame))
            colour = QColor(EDIT_COLOUR if origin == "edit" else PROPOSAL_COLOURS[tr.proposal.confidence])
            chosen = tr is self.selection
            p.setPen(QPen(QColor("#ffe040") if chosen else colour, 3 if chosen else 2)); p.setBrush(Qt.NoBrush)
            p.drawEllipse(pt, 11, 11)
            p.setPen(QColor("#ffffff"))
            p.drawText(int(pt.x()) + 14, int(pt.y()) - 6, slot or "?")
            self._rings.append((pt, tr))

    def _ring_at(self, pos) -> Tracklet | None:
        best = min(((abs(pt.x() - pos.x()) + abs(pt.y() - pos.y()), tr) for pt, tr in self._rings), default=None,
                   key=lambda r: r[0])
        return best[1] if best and best[0] < 26 else None

    def mousePressEvent(self, e) -> None:
        tr = self._ring_at(e.position())
        if tr is None:
            return
        self.tracklet_picked.emit(tr)
        if e.button() == Qt.RightButton:
            self.context_requested.emit(tr, e.globalPosition().toPoint())


class ReviewWindow(QMainWindow):
    def __init__(self, data: ReviewData) -> None:
        super().__init__()
        self.data = data
        self.setWindowTitle("Dot assignment review")
        cameras = [{"shot_video_id": data.svid_by_cam[c], "file_path": data.videos[c]} for c in data.tracklets]
        self.service = FrameService(cameras)
        self.service.frame_ready.connect(lambda *_: self.frame.update())
        for cam in data.tracklets:               # a decoder needs a first request before it can run
            self.service.request(data.svid_by_cam[cam], 0)
        self.sheet, self.frame = DopeSheet(data), None
        self.frame = FramePane(data, self.service)
        split = QSplitter(); split.addWidget(self.sheet); split.addWidget(self.frame)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        bar = QHBoxLayout()
        self.camera_box = QComboBox()
        for cam in data.tracklets:
            self.camera_box.addItem(data.labels.get(cam, cam), cam)
        self.apply_box = QComboBox()
        self.apply_box.addItems(["whole tracklet", "from playhead on", "until playhead"])
        none_box = QCheckBox("show unassigned")
        none_box.setChecked(True)
        next_btn = QPushButton("Next to review (N)")
        accept_btn = QPushButton("Accept (A)")
        accept_all = QPushButton("Accept all high confidence")
        for w in (QLabel("Camera:"), self.camera_box, QLabel("Edit applies to:"), self.apply_box, none_box,
                  next_btn, accept_btn, accept_all):
            bar.addWidget(w)
        bar.addStretch()
        self.status = QLabel()
        root = QWidget(); lay = QVBoxLayout(root); lay.addLayout(bar); lay.addWidget(split, 1); lay.addWidget(self.status)
        self.setCentralWidget(root)
        self.resize(1500, 900)
        self.camera_box.currentIndexChanged.connect(self._camera_changed)
        none_box.toggled.connect(self._toggle_none)
        next_btn.clicked.connect(self.next_to_review)
        accept_btn.clicked.connect(self.accept_selected)
        accept_all.clicked.connect(self.accept_all_high)
        self.sheet.selected.connect(self._selected)
        self.sheet.time_picked.connect(self._time)
        self.sheet.context_requested.connect(self.context_menu)
        self.frame.tracklet_picked.connect(self._pick_from_frame)
        self.frame.context_requested.connect(self.context_menu)
        for key, fn in (("N", self.next_to_review), ("A", self.accept_selected), ("Delete", self.reject_selected),
                        ("Ctrl+Z", self.undo), ("Z", self.toggle_zoom), ("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1))):
            QShortcut(QKeySequence(key), self, activated=fn)
        self.sheet.playhead = self.sheet.t_left + 0.5
        self._refresh_status()

    @property
    def camera(self) -> str:
        return self.camera_box.currentData()

    def closeEvent(self, e) -> None:
        self.service.shutdown()
        super().closeEvent(e)

    def _camera_changed(self) -> None:
        self.sheet.selection = None
        self.sheet.set_camera(self.camera)
        self.frame.show_at(self.camera, self.sheet.playhead, None)

    def _toggle_none(self, on: bool) -> None:
        self.sheet.show_none = on
        self.sheet.refresh()

    def _selected(self, t: Tracklet | None) -> None:
        self.frame.show_at(self.camera, self.sheet.playhead, t)

    def _pick_from_frame(self, t: Tracklet) -> None:
        self.sheet.select(t)
        self.sheet.reveal(t)

    def _time(self, t: float) -> None:
        self.sheet.playhead = t
        self.frame.show_at(self.camera, t, self.sheet.selection)
        self.sheet.update()

    def step(self, n: int) -> None:
        frame = self.data.frame_at(self.camera, self.sheet.playhead) + n
        self._time(self.data.time_of(self.camera, frame))

    def _range(self, t: Tracklet) -> tuple[int | None, int | None]:
        frame = self.data.frame_at(self.camera, self.sheet.playhead)
        mode = self.apply_box.currentIndex()
        return (None, None) if mode == 0 else (frame, t.last_frame) if mode == 1 else (t.first_frame, frame)

    def _changed(self) -> None:
        self.sheet.refresh(); self.frame.update(); self._refresh_status()

    def _refresh_status(self) -> None:
        mine = list(self.data.tracklets[self.camera].values())
        todo = sum(1 for t in mine if self.data.needs_review(t))
        done = sum(1 for t in mine if self.data.edits_of(t))
        self.status.setText(f"{self.camera_box.currentText()}: {len(mine)} tracklets, {done} with a decision, {todo} left to review")

    def toggle_zoom(self) -> None:
        self.frame.zoomed = not self.frame.zoomed
        self.frame.update()

    def accept_selected(self) -> None:
        t = self.sheet.selection
        if t is not None and t.proposal.slot is not None and not self.data.edits_of(t):
            self.data.add(t, "assign", t.proposal.slot)
            self._changed()

    def reject_selected(self) -> None:
        t = self.sheet.selection
        if t is not None:
            self.data.add(t, "reject", None, *self._range(t))
            self._changed()

    def accept_all_high(self) -> None:
        for t in self.data.tracklets[self.camera].values():
            if t.proposal.confidence == "high" and t.proposal.slot and not self.data.edits_of(t):
                self.data.add(t, "assign", t.proposal.slot)
        self._changed()

    def undo(self) -> None:
        if self.data.undo_last():
            self._changed()

    def next_to_review(self) -> None:
        mine = sorted((t for t in self.data.tracklets[self.camera].values() if self.data.needs_review(t)),
                      key=lambda t: self.data.time_of(self.camera, t.first_frame))
        after = self.data.time_of(self.camera, self.sheet.selection.first_frame) if self.sheet.selection else -1.0
        nxt = next((t for t in mine if self.data.time_of(self.camera, t.first_frame) > after + 1e-6), mine[0] if mine else None)
        if nxt is None:
            return
        self.sheet.select(nxt)
        self.sheet.reveal(nxt)
        mid = (nxt.first_frame + nxt.last_frame) // 2
        self._time(self.data.time_of(self.camera, mid))
        self.frame.show_at(self.camera, self.sheet.playhead, nxt)

    def context_menu(self, t: Tracklet, pos) -> None:
        menu = QMenu(self)
        frame = self.data.frame_at(self.camera, self.sheet.playhead)
        current = self.data.slot_at(t, min(max(frame, t.first_frame), t.last_frame))[0]
        first, last = self._range(t)
        if t.proposal.slot and not self.data.edits_of(t):
            menu.addAction(f"Accept {t.proposal.slot}", lambda: (self.data.add(t, "assign", t.proposal.slot), self._changed()))
        assign = menu.addMenu("Assign to")
        group = self.data.group_of.get(current or t.proposal.slot)
        near = [g for g in (group, mirror_group(group) if group else None) if g in self.data.groups]
        for g in dict.fromkeys(near):
            for s in self.data.groups[g]:
                assign.addAction(s, lambda s=s: (self.data.add(t, "assign", s, first, last), self._changed()))
            assign.addSeparator()
        for g, slots in self.data.groups.items():
            if g in near:
                continue
            sub = assign.addMenu(g)
            for s in slots:
                sub.addAction(s, lambda s=s: (self.data.add(t, "assign", s, first, last), self._changed()))
        if current:
            menu.addAction(f"Not {current} (forbid)", lambda: (self.data.add(t, "forbid", current, first, last), self._changed()))
        menu.addAction("Not a marker (reject)", lambda: (self.data.add(t, "reject", None, first, last), self._changed()))
        if self.data.edits_of(t):
            menu.addAction("Clear my edits", lambda: (self.data.clear(t), self._changed()))
        menu.exec(pos)


def main() -> int:
    from posetrak.db.db import open_session

    args = [a for a in sys.argv[1:]]
    if len(args) != 4:
        print(__doc__)
        return 2
    session_db, trial, detection_run, attachment = args
    import sqlite3
    conn = open_session(Path(session_db))
    conn.row_factory = sqlite3.Row
    trial = conn.execute("SELECT id FROM trials WHERE id LIKE ?", (trial + "%",)).fetchone()[0]
    detection_run = conn.execute("SELECT id FROM detection_runs WHERE id LIKE ?", (detection_run + "%",)).fetchone()[0]
    app = QApplication(sys.argv)
    window = ReviewWindow(ReviewData(conn, trial, detection_run, attachment))
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
