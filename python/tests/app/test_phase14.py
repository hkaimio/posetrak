"""Tests for Phase 14: multi-keyframe interpolation.

Generalizes Phase 10's two-anchor `I` interpolation to N anchors: any
interior frame in the active range that is an explicit keyframe for a given
keypoint (an edit row with is_outlier == 0, i.e. STATUS_BLUE) also anchors
the interpolation, splitting it into independent piecewise segments. An
untouched original detection — even a currently-inlier one — is never an
interior anchor, so "select a wide range, press I" still overwrites
everything between the two ends with a single straight line when the user
hasn't deliberately kept any interior frame.

See docs/roadmap/features/keypoint-editing/keypoint-editing-design.md,
"Multi-keyframe interpolation", for the three validation cases this file
covers.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider

_N_KP = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _enc(x: float, y: float, conf: float) -> bytes:
    kp = np.zeros((_N_KP, 3), dtype=np.float32)
    kp[0] = [x, y, conf]
    kp[1] = [x, y, conf]
    return kp.tobytes()


@pytest.fixture()
def multikey_db(tmp_path):
    """Observations at frames 1, 10, 20 (30fps): a straight line 1->20 would
    put frame 10 at (10, 10), but its actual detection is (99, 99) — a wrong,
    untouched inlier, exactly the "plain overwrite" scenario from the brief.
    """
    from posetrak.db.db import create_session
    conn = create_session(tmp_path / "multikey.db")
    conn.execute("PRAGMA foreign_keys = OFF")

    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('s1', '2026-01-01T00:00:00Z')")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('sh1', 's1', 1)")
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci1', 'cm1', 'A')")
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv1', 'sh1', 'ci1', '/dev/null', 0, 100, 30.0)"
    )
    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id, pixels_are_undistorted)"
        " VALUES ('seq1', 'sh1', 'sc1', 0.0, 2.0, 'run1', 0)"
    )
    conn.execute("INSERT INTO sequence_persons (sequence_id, person_id, person_name)"
                 " VALUES ('seq1', 0, 'Alice')")

    for frame, (x, y, c) in {
        1: (1.0, 1.0, 0.9),
        10: (99.0, 99.0, 0.9),
        20: (20.0, 20.0, 0.9),
    }.items():
        conn.execute(
            "INSERT INTO pose_observations"
            " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
            " VALUES ('seq1', 'ci1', ?, ?, 0, ?)",
            (frame, frame / 30.0, _enc(x, y, c)),
        )
    conn.commit()
    yield conn
    conn.close()


def _make_widget(qapp, db):
    from app.ui.content_panels import PersonCropGridWidget
    from PySide6.QtWidgets import QWidget

    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
    QWidget.__init__(w)
    w._conn = db
    w._sequence_id = "seq1"
    w._cells = []
    w._cameras = [{"shot_video_id": "sv1", "camera_instance_id": "ci1", "label": "A"}]
    w._t_start = 0.0
    w._t_end = 2.0
    w._current_t = 0.0
    w._slider = QSlider(Qt.Orientation.Horizontal)
    w._slider.setMinimum(0)
    w._slider.setMaximum(2000)
    w._slider.setSingleStep(1)  # dense sampling so every integer frame 1-20 is captured
    w._slider.setValue(0)
    w._time_label = None
    w._show_detected = None
    w._show_tracked = None
    w._show_seg = None
    w._edit_btn = None
    w._edit_mode = True
    w._sel_kp_indices = {0}
    w._primary_kp_idx = 0
    w._sel_cam_idx = 0
    w._obs_kp = {}
    w._det_bboxes = {}
    w._marker_proj = {}
    w._joint_proj = {}
    w._bone_pairs = []
    w._tracking_timestamps = []
    w._outlier_masks = {}
    w._hand_redetect = None
    w._hand_redetect_timers = {}
    w._seg_sources = {}
    w._track_segs = {}
    w._video_dims = {}
    w._3d_ph = None
    w._ncols = 1
    w._grid = None
    w._backfill = None
    w._clipboard = None
    w._clipboard_cam_idx = None
    w._range_start_v = 33     # round(1/30 * 1000)  -> frame 1
    w._range_end_v = 667      # round(20/30 * 1000) -> frame 20
    w._timeline = None
    w._timeline_status_by_cam = {}
    w._timeline_inlier_counts = {}
    w._load_frame = MagicMock()

    from app.pose.db_cache import read_observations_with_edits
    w._obs_kp["ci1"] = read_observations_with_edits(db, "seq1", "ci1")

    mock_sync = MagicMock()
    mock_sync.lookup = lambda t, svid: round(t * 30)
    w._sync_table = mock_sync

    return w


def _edit_xy(db, frame: int, kp_idx: int = 0) -> tuple[float, float] | None:
    """Return the edited (x, y) for kp_idx at frame, or None if that slot's
    kp_mask bit isn't set (no override) — mirrors what the merge logic
    actually treats as "edited", not just whatever zero-initialized value
    happens to sit in an unrelated slot of the same blob."""
    row = db.execute(
        "SELECT kp_blob, kp_mask FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=?",
        (frame,),
    ).fetchone()
    if row is None:
        return None
    mask = bytes(row["kp_mask"])
    byte_idx, bit_idx = divmod(kp_idx, 8)
    if byte_idx >= len(mask) or not ((mask[byte_idx] >> bit_idx) & 1):
        return None
    kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    return float(kp[kp_idx, 0]), float(kp[kp_idx, 1])


# ---------------------------------------------------------------------------
# Validation case 1: plain overwrite — no interior keyframes means the
# untouched (but wrong) inlier at frame 10 must NOT act as an anchor.
# ---------------------------------------------------------------------------

def test_plain_overwrite_ignores_untouched_interior_inlier(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)

    w._interpolate_range()

    # Frame 10 should be overwritten to the straight-line value (10, 10),
    # not left at its original wrong detection (99, 99).
    x, y = _edit_xy(multikey_db, 10)
    assert x == pytest.approx(10.0, abs=0.1)
    assert y == pytest.approx(10.0, abs=0.1)


def test_plain_overwrite_is_a_single_straight_line(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)

    w._interpolate_range()

    for frame in (2, 5, 9, 11, 15, 19):
        x, y = _edit_xy(multikey_db, frame)
        assert x == pytest.approx(float(frame), abs=0.1), f"frame {frame}"
        assert y == pytest.approx(float(frame), abs=0.1), f"frame {frame}"


def test_plain_overwrite_does_not_touch_endpoints(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)

    w._interpolate_range()

    assert _edit_xy(multikey_db, 1) is None
    assert _edit_xy(multikey_db, 20) is None


# ---------------------------------------------------------------------------
# Validation case 2: multi-keyframe — an explicit interior keyframe (moved,
# not just re-enabled at its original spot) splits the interpolation into
# two independent segments.
# ---------------------------------------------------------------------------

def test_multikeyframe_moved_interior_anchor_splits_interpolation(qapp, multikey_db):
    from app.pose.db_cache import update_single_keypoint_edit
    # Frame 10 explicitly re-enabled/moved to (10, 10) — distinct from its
    # original wrong detection (99, 99).
    update_single_keypoint_edit(multikey_db, "seq1", "ci1", 10, kp_idx=0,
                                 new_x=10.0, new_y=10.0, is_outlier=False)

    w = _make_widget(qapp, multikey_db)
    w._interpolate_range()

    # Segment 1->10 and segment 10->20 are each a straight line with slope 1,
    # so frame 5 and frame 15 land exactly on (5, 5) and (15, 15) either way
    # — the real test is that frame 10 itself is untouched by interpolation.
    assert _edit_xy(multikey_db, 5) == pytest.approx((5.0, 5.0), abs=0.1)
    assert _edit_xy(multikey_db, 15) == pytest.approx((15.0, 15.0), abs=0.1)
    assert _edit_xy(multikey_db, 10) == pytest.approx((10.0, 10.0), abs=1e-4)


def test_multikeyframe_anchor_frame_not_rewritten(qapp, multikey_db):
    """The interior anchor's edit row must not be touched by interpolation —
    verified by checking it keeps exactly the value we set, not some blended
    or re-derived one."""
    from app.pose.db_cache import update_single_keypoint_edit
    update_single_keypoint_edit(multikey_db, "seq1", "ci1", 10, kp_idx=0,
                                 new_x=12.5, new_y=-3.5, is_outlier=False)

    w = _make_widget(qapp, multikey_db)
    w._interpolate_range()

    assert _edit_xy(multikey_db, 10) == pytest.approx((12.5, -3.5), abs=1e-4)


# ---------------------------------------------------------------------------
# Validation case 3: Ctrl+click keyframe — freezing a frame at its *current*
# (unchanged) value still makes it an anchor.
# ---------------------------------------------------------------------------

def test_ctrl_click_freeze_at_original_position_still_anchors(qapp, multikey_db):
    """Frame 10 is "frozen" via an edit row that writes the *same* (99, 99)
    position it already had — mirroring what _on_timeline_keyframe_toggle's
    freeze path does. The point isn't that the value changed, it's that an
    edit row with is_outlier=0 now exists at that frame."""
    from app.pose.db_cache import update_single_keypoint_edit
    update_single_keypoint_edit(multikey_db, "seq1", "ci1", 10, kp_idx=0,
                                 new_x=99.0, new_y=99.0, is_outlier=False)

    w = _make_widget(qapp, multikey_db)
    w._interpolate_range()

    # Frame 5 is now interpolated toward (99, 99) at frame 10, not toward the
    # straight-line (10, 10) — proof that frame 10 split the interpolation.
    x5, y5 = _edit_xy(multikey_db, 5)
    assert x5 == pytest.approx(44.56, abs=0.5)
    assert y5 == pytest.approx(44.56, abs=0.5)
    # Frame 10 itself is unchanged.
    assert _edit_xy(multikey_db, 10) == pytest.approx((99.0, 99.0), abs=1e-4)


# ---------------------------------------------------------------------------
# Supporting behavior
# ---------------------------------------------------------------------------

def test_disabled_interior_frame_does_not_anchor(qapp, multikey_db):
    """A frame marked outlier (disabled) has an edit row too, but with
    is_outlier=1 — it must not anchor the interpolation either."""
    from app.pose.db_cache import update_single_keypoint_edit
    update_single_keypoint_edit(multikey_db, "seq1", "ci1", 10, kp_idx=0,
                                 new_x=99.0, new_y=99.0, is_outlier=True)

    w = _make_widget(qapp, multikey_db)
    w._interpolate_range()

    # Straight line 1->20 still wins; frame 10 gets overwritten to (10, 10).
    assert _edit_xy(multikey_db, 10) == pytest.approx((10.0, 10.0), abs=0.1)


def test_multiple_interior_anchors_create_three_segments(qapp, multikey_db):
    from app.pose.db_cache import update_single_keypoint_edit
    update_single_keypoint_edit(multikey_db, "seq1", "ci1", 6, kp_idx=0,
                                 new_x=100.0, new_y=100.0, is_outlier=False)
    update_single_keypoint_edit(multikey_db, "seq1", "ci1", 14, kp_idx=0,
                                 new_x=200.0, new_y=200.0, is_outlier=False)

    w = _make_widget(qapp, multikey_db)
    w._interpolate_range()

    # Segment 1: frame 1 (1,1) -> frame 6 (100,100). Frame 3: t=2/5.
    x3, y3 = _edit_xy(multikey_db, 3)
    assert x3 == pytest.approx(1 + 2 / 5 * 99, abs=0.5)
    # Segment 3: frame 14 (200,200) -> frame 20 (20,20). Frame 17: t=3/6.
    x17, y17 = _edit_xy(multikey_db, 17)
    assert x17 == pytest.approx(200 + 0.5 * (20 - 200), abs=0.5)
    # Anchors themselves untouched.
    assert _edit_xy(multikey_db, 6) == pytest.approx((100.0, 100.0), abs=1e-4)
    assert _edit_xy(multikey_db, 14) == pytest.approx((200.0, 200.0), abs=1e-4)


def test_unselected_keypoint_index_is_not_interpolated(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._sel_kp_indices = {0}  # kp_idx=1 not selected

    w._interpolate_range()

    assert _edit_xy(multikey_db, 10, kp_idx=1) is None


def test_low_confidence_endpoint_skips_keypoint_entirely(qapp, multikey_db):
    """Existing Phase 10 guard: if either range boundary has near-zero
    confidence for a keypoint, that keypoint is skipped — must still hold
    with the anchor-scan added on top."""
    from app.pose.db_cache import update_single_keypoint_edit
    update_single_keypoint_edit(multikey_db, "seq1", "ci1", 1, kp_idx=0,
                                 new_x=1.0, new_y=1.0, is_outlier=True)

    w = _make_widget(qapp, multikey_db)
    w._interpolate_range()

    assert _edit_xy(multikey_db, 10) is None


def test_interpolate_clears_range_and_calls_load_frame(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._interpolate_range()
    assert w._range_start_v is None
    assert w._range_end_v is None
    w._load_frame.assert_called_once()


# ---------------------------------------------------------------------------
# Regression: interpolating must not reset the timeline's zoom/pan.
#
# _interpolate_range refreshes the edited camera's timeline status
# (_refresh_timeline_status -> _push_timeline_camera_data), which used to
# call KeypointTimelineWidget.set_time_range unconditionally — that resets
# the visible time window to the full trial on every single edit, silently
# undoing whatever the user had zoomed into.
# ---------------------------------------------------------------------------

def test_interpolate_does_not_reset_timeline_zoom(qapp, multikey_db):
    from app.pose.kp_models import COCO17
    from app.ui.keypoint_timeline_widget import KeypointTimelineWidget

    w = _make_widget(qapp, multikey_db)
    w._timeline = KeypointTimelineWidget(COCO17, w._cameras)
    w._timeline.set_time_range(w._t_start, w._t_end, "sv1", w._sync_table)
    w._timeline._canvas.resize(340, w._timeline._canvas.minimumHeight())
    w._timeline._canvas.set_current_time_v(300)
    w._timeline._canvas.zoom(0.3)  # zoom in around ms=300
    zoomed_view = w._timeline._canvas.view_range()
    assert zoomed_view != (0, w._timeline._canvas.total_ms())  # sanity: really zoomed

    w._interpolate_range()

    assert w._timeline._canvas.view_range() == zoomed_view
