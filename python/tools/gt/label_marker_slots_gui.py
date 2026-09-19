# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""label_marker_slots_gui.py — PySide6 GUI for tagging detected reflective
dots with a marker **slot** name, to build the ground truth the
marker-assignment metric harness needs (redesign doc §8 P-C).

Supersedes label_dot_ground_truth.py's `--slots` mode: the cv2 tool has
no context menus and every left-click adds a point, so accidental clicks
pile up and re-labelling means delete+re-click. Here:

  * a frame opens with the detector's own dot candidates already placed
    (grey = untagged);
  * **right-click a dot** -> menu -> pick its slot (or `unlabeled`, or
    Delete);
  * **Ctrl + left-click empty space** -> add a dot the detector missed;
  * left-click selects; number keys 1-9 / 0 set the selected dot's slot;
    Delete removes it;
  * the previous same-camera frame's tagged dots show as faint ghosts;
    "Carry from prev" snaps each to the nearest current dot.

**Track id is derived, not entered**: one physical marker per slot, so a
tagged dot's track is `slot_palette.index(slot)` (stable per slot;
downstream metrics key on (camera, track), so a value shared across
cameras is fine). `unlabeled` dots get no track.

Output JSON is the same shape label_dot_ground_truth.py writes -- a list
of frame entries with `points: [[x, y], ...]` and a parallel
`point_meta: [{"slot": ..., "track": ...}, ...]` -- so
refine_dot_ground_truth.py and validate_dot_detector.py consume it
unchanged.

Usage:
    python tools/label_marker_slots_gui.py \\
        --manifest scratch/dot_ground_truth/pc_tiny_manifest.json \\
        --session nelli=/path/to/session.db \\
        --seed-detection-run 01c2e3c1-204b-49f8-9b14-6fd611b765a4 \\
        --slots "hip_R,knee_lat_R,knee_med_R,knee_front_R,ankle_lat_R,ankle_med_R,heel_R,toe_R,hip_L,knee_lat_L,knee_med_L,knee_front_L,ankle_lat_L,ankle_med_L,heel_L,toe_L" \\
        --output scratch/dot_ground_truth/pc_labels.json
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402

_UNLABELED = "unlabeled"
_GHOST_SNAP_PX = 40.0


def _key(entry: dict) -> str:
    return f"{entry['dataset']}::{entry['camera_label']}::{entry['video_frame']}"


def _slot_qcolor(slot: str) -> QtGui.QColor:
    if not slot or slot == _UNLABELED:
        return QtGui.QColor(170, 170, 170)
    h = (abs(hash(slot)) % 997) / 997.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return QtGui.QColor(int(r * 255), int(g * 255), int(b * 255))


def _bgr_to_qpixmap(img: np.ndarray) -> QtGui.QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qim = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qim)


def _parse_slots(spec: str) -> list[str]:
    if spec.startswith("@"):
        names = [ln.strip() for ln in Path(spec[1:]).read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
    if _UNLABELED not in names:
        names.append(_UNLABELED)
    return names


class _PointItem(QtWidgets.QGraphicsEllipseItem):
    _R = 7.0

    def __init__(self, x: float, y: float, slot: str = _UNLABELED) -> None:
        super().__init__(-self._R, -self._R, 2 * self._R, 2 * self._R)
        self.setPos(x, y)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(10)
        self._label = QtWidgets.QGraphicsSimpleTextItem("", self)
        self._label.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self._label.setPos(self._R + 2, -self._R - 2)
        f = self._label.font()
        f.setPointSize(9)
        self._label.setFont(f)
        self.slot = slot
        self.set_slot(slot)

    def set_slot(self, slot: str) -> None:
        self.slot = slot
        col = _slot_qcolor(slot)
        tagged = slot and slot != _UNLABELED
        self.setPen(QtGui.QPen(col, 2))
        self.setBrush(QtGui.QBrush(col if tagged else QtCore.Qt.NoBrush))
        self._label.setText(slot if tagged else "")
        self._label.setBrush(QtGui.QBrush(col))

    def set_selected_look(self, on: bool) -> None:
        self.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0) if on else _slot_qcolor(self.slot),
                               3 if on else 2))


class _GhostItem(QtWidgets.QGraphicsEllipseItem):
    _R = 6.0

    def __init__(self, x: float, y: float, slot: str) -> None:
        super().__init__(-self._R, -self._R, 2 * self._R, 2 * self._R)
        self.setPos(x, y)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(5)
        col = _slot_qcolor(slot)
        col.setAlpha(110)
        self.setPen(QtGui.QPen(col, 1, QtCore.Qt.DashLine))
        t = QtWidgets.QGraphicsSimpleTextItem(slot, self)
        t.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        t.setPos(self._R + 2, self._R)
        t.setBrush(QtGui.QBrush(col))
        self.slot = slot


class _Canvas(QtWidgets.QGraphicsView):
    changed = QtCore.Signal()

    def __init__(self, slot_palette: list[str]) -> None:
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self._slot_palette = slot_palette
        self._pix_item: QtWidgets.QGraphicsPixmapItem | None = None
        self.points: list[_PointItem] = []
        self.ghosts: list[_GhostItem] = []
        self.selected: _PointItem | None = None

    # ---- frame lifecycle ----
    def set_frame(self, pixmap: QtGui.QPixmap,
                  seeded: list[tuple[float, float, str]],
                  ghosts: list[tuple[float, float, str]]) -> None:
        self._scene.clear()
        self.points.clear()
        self.ghosts.clear()
        self.selected = None
        self._pix_item = self._scene.addPixmap(pixmap)
        self._pix_item.setZValue(0)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        for gx, gy, gslot in ghosts:
            g = _GhostItem(gx, gy, gslot)
            self._scene.addItem(g)
            self.ghosts.append(g)
        for x, y, slot in seeded:
            self._add_point(x, y, slot)
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def _add_point(self, x: float, y: float, slot: str = _UNLABELED) -> _PointItem:
        it = _PointItem(x, y, slot)
        self._scene.addItem(it)
        self.points.append(it)
        return it

    def export_points(self) -> tuple[list[list[float]], list[dict]]:
        pts, meta = [], []
        for it in self.points:
            p = it.pos()
            pts.append([float(p.x()), float(p.y())])
            slot = it.slot
            track = self._slot_palette.index(slot) if slot and slot != _UNLABELED else None
            meta.append({"slot": slot, "track": track})
        return pts, meta

    # ---- interaction ----
    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def _point_at(self, view_pos: QtCore.QPoint) -> _PointItem | None:
        for it in self.itemAt(view_pos), *self.items(view_pos):
            if isinstance(it, _PointItem):
                return it
        return None

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        scene_pos = self.mapToScene(ev.position().toPoint())
        hit = self._point_at(ev.position().toPoint())
        if ev.button() == QtCore.Qt.RightButton and hit is not None:
            self._slot_menu(hit, ev.globalPosition().toPoint())
            return
        if ev.button() == QtCore.Qt.LeftButton and (ev.modifiers() & QtCore.Qt.ControlModifier):
            it = self._add_point(scene_pos.x(), scene_pos.y(), _UNLABELED)
            self._select(it)
            self.changed.emit()
            return
        if ev.button() == QtCore.Qt.LeftButton and hit is not None:
            self._select(hit)
            return
        super().mousePressEvent(ev)

    def _select(self, it: _PointItem | None) -> None:
        if self.selected is not None:
            self.selected.set_selected_look(False)
        self.selected = it
        if it is not None:
            it.set_selected_look(True)

    def _slot_menu(self, item: _PointItem, global_pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        for name in self._slot_palette:
            act = menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(name == item.slot)
            act.triggered.connect(lambda _c=False, n=name, i=item: (i.set_slot(n), self.changed.emit()))
        menu.addSeparator()
        dele = menu.addAction("Delete point")
        dele.triggered.connect(lambda: self._delete(item))
        menu.exec(global_pos)

    def _delete(self, item: _PointItem) -> None:
        if self.selected is item:
            self.selected = None
        self._scene.removeItem(item)
        self.points.remove(item)
        self.changed.emit()

    def set_selected_slot_by_index(self, digit: int) -> None:
        if self.selected is None:
            return
        idx = 9 if digit == 0 else digit - 1  # 1..9 -> 0..8, 0 -> last (unlabeled) handled below
        if digit == 0:
            self.selected.set_slot(_UNLABELED)
        elif idx < len(self._slot_palette):
            self.selected.set_slot(self._slot_palette[idx])
        self.changed.emit()

    def delete_selected(self) -> None:
        if self.selected is not None:
            self._delete(self.selected)

    def carry_from_prev(self) -> None:
        for g in self.ghosts:
            gp = g.pos()
            best, best_d = None, None
            for it in self.points:
                if it.slot and it.slot != _UNLABELED:
                    continue
                d = (it.pos() - gp).manhattanLength()
                if best_d is None or d < best_d:
                    best, best_d = it, d
            if best is not None and best_d is not None and best_d <= _GHOST_SNAP_PX:
                best.set_slot(g.slot)
        self.changed.emit()


class _MainWindow(QtWidgets.QMainWindow):
    def __init__(self, manifest: list[dict], labels: dict, slot_palette: list[str],
                 session_conns: dict, seed_run: str | None, out_path: Path) -> None:
        super().__init__()
        self.manifest = manifest
        self.labels = labels
        self.slot_palette = slot_palette
        self.session_conns = session_conns
        self.seed_run = seed_run
        self.out_path = out_path
        self.idx = 0
        self.dirty = False

        self.canvas = _Canvas(slot_palette)
        self.canvas.changed.connect(self._on_changed)
        self.setCentralWidget(self.canvas)

        tb = self.addToolBar("nav")
        tb.addAction("◀ Prev", self.prev)
        tb.addAction("Next ▶", self.next)
        tb.addAction("Carry from prev", self.canvas.carry_from_prev)
        tb.addAction("Fit", lambda: self.canvas.fitInView(
            self.canvas.sceneRect(), QtCore.Qt.KeepAspectRatio))
        tb.addAction("Save", self.save_current)

        QtGui.QShortcut(QtGui.QKeySequence("PgDown"), self, self.next)
        QtGui.QShortcut(QtGui.QKeySequence("PgUp"), self, self.prev)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self.save_current)
        QtGui.QShortcut(QtGui.QKeySequence("Delete"), self, self.canvas.delete_selected)
        for d in range(10):
            QtGui.QShortcut(QtGui.QKeySequence(str(d)), self,
                            lambda dd=d: self.canvas.set_selected_slot_by_index(dd))

        self.statusBar()
        self.load_frame(0)

    # ---- data ----
    def _cam_key(self, e: dict) -> tuple[str, str]:
        return (e["dataset"], e["camera_label"])

    def _ghosts_for(self, i: int) -> list[tuple[float, float, str]]:
        want = self._cam_key(self.manifest[i])
        for j in range(i - 1, -1, -1):
            if self._cam_key(self.manifest[j]) != want:
                continue
            e = self.labels.get(_key(self.manifest[j]))
            if not e or not e.get("points"):
                return []
            pm = e.get("point_meta", [])
            out = []
            for k, xy in enumerate(e["points"]):
                slot = pm[k].get("slot") if k < len(pm) else None
                if slot and slot != _UNLABELED:
                    out.append((float(xy[0]), float(xy[1]), slot))
            return out
        return []

    def load_frame(self, i: int) -> None:
        self.idx = i
        e = self.manifest[i]
        conn = self.session_conns[e["dataset"]]
        fp = conn.execute("SELECT file_path FROM capture_videos WHERE id = ?",
                          (e["shot_video_id"],)).fetchone()[0]
        img = None
        for _, dec in iter_frames(fp, e["video_frame"], e["video_frame"] + 1):
            img = dec
            break
        if img is None:
            raise SystemExit(f"could not decode frame {e['video_frame']} from {fp}")

        key = _key(e)
        saved = self.labels.get(key)
        seeded: list[tuple[float, float, str]] = []
        if saved and "points" in saved:
            pm = saved.get("point_meta", [])
            for k, xy in enumerate(saved["points"]):
                slot = pm[k].get("slot") if k < len(pm) else _UNLABELED
                seeded.append((float(xy[0]), float(xy[1]), slot or _UNLABELED))
        elif self.seed_run:
            row = conn.execute(
                "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? "
                "AND shot_video_id = ? AND video_frame = ? AND region_type = 'dots'",
                (self.seed_run, e["shot_video_id"], e["video_frame"]),
            ).fetchone()
            if row is not None:
                for cx, cy in decode_dot_candidates(bytes(row[0]))[:, :2]:
                    seeded.append((float(cx), float(cy), _UNLABELED))

        self.canvas.set_frame(_bgr_to_qpixmap(img), seeded, self._ghosts_for(i))
        self.dirty = False
        self._refresh_status()

    def save_current(self) -> None:
        e = dict(self.manifest[self.idx])
        pts, meta = self.canvas.export_points()
        e["points"] = pts
        e["point_meta"] = meta
        self.labels[_key(e)] = e
        ordered = sorted(self.labels.values(),
                         key=lambda x: (x["dataset"], x["camera_label"], x["video_frame"]))
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(ordered, indent=2))
        self.dirty = False
        self._refresh_status()

    def _on_changed(self) -> None:
        self.dirty = True
        self._refresh_status()

    def _refresh_status(self) -> None:
        e = self.manifest[self.idx]
        _pts, meta = self.canvas.export_points()
        tagged = sorted({m["slot"] for m in meta if m["slot"] and m["slot"] != _UNLABELED})
        done = sum(1 for m in self.manifest if _key(m) in self.labels and self.labels[_key(m)].get("points"))
        self.setWindowTitle(f"label_marker_slots — [{self.idx + 1}/{len(self.manifest)}] "
                            f"{e['camera_label']} f={e['video_frame']}"
                            + ("  *unsaved*" if self.dirty else ""))
        self.statusBar().showMessage(
            f"dots: {len(_pts)}   tagged slots ({len(tagged)}/{len(self.slot_palette) - 1}): "
            f"{', '.join(tagged)}   |   frames labelled: {done}/{len(self.manifest)}")

    def prev(self) -> None:
        if self.dirty:
            self.save_current()
        if self.idx > 0:
            self.load_frame(self.idx - 1)

    def next(self) -> None:
        if self.dirty:
            self.save_current()
        if self.idx < len(self.manifest) - 1:
            self.load_frame(self.idx + 1)

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        if self.dirty:
            self.save_current()
        ev.accept()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--session", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--output", required=True)
    ap.add_argument("--slots", required=True, help="Comma-separated slot names, or @path to a file.")
    ap.add_argument("--seed-detection-run", default=None)
    args = ap.parse_args()

    slot_palette = _parse_slots(args.slots)
    session_paths = {}
    for entry in args.session:
        name, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--session must be NAME=PATH, got {entry!r}")
        session_paths[name] = path
    session_conns = {n: sqlite3.connect(f"file:{p}?mode=ro", uri=True) for n, p in session_paths.items()}

    manifest = json.loads(Path(args.manifest).read_text())
    out_path = Path(args.output)
    labels = {}
    if out_path.exists():
        labels = {_key(e): e for e in json.loads(out_path.read_text())}

    app = QtWidgets.QApplication(sys.argv)
    win = _MainWindow(manifest, labels, slot_palette, session_conns, args.seed_detection_run, out_path)
    win.resize(1500, 1000)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
