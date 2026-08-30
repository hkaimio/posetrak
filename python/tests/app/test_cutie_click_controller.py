# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ClickController's manual paint/erase overlay
(segmentation-ui-improvements design doc, Issue 5) -- the compositing
layer applied after base_mask + SAM2, always winning, so manual edits
survive _run_predictions()'s per-click rebuild.

Uses a bogus model name so construction doesn't try to load a real SAM2
checkpoint -- these tests exercise the overlay/compositing logic, which
works identically whether or not SAM2 itself is available (self.available
is False either way, so _run_predictions() never actually calls into it).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.pose.cutie_click_controller import PAINT_UNTOUCHED, ClickController


@pytest.fixture()
def ctrl():
    c = ClickController(model_name="nonexistent/model-for-tests")
    assert not c.available  # sanity: no real SAM2 involved
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    c.set_image(frame)
    return c


def test_paint_circle_stamps_label_into_mask(ctrl):
    mask = ctrl.paint_circle(1, 40, 40, radius=10)
    assert mask[40, 40] == 1
    assert mask[0, 0] == 0  # untouched, far outside the circle


def test_erase_circle_stamps_background(ctrl):
    ctrl.paint_circle(1, 40, 40, radius=15)
    mask = ctrl.erase_circle(40, 40, radius=5)
    assert mask[40, 40] == 0
    # Just outside the erased circle but inside the original paint --
    # still label 1 (erase only clears its own smaller circle).
    assert mask[40, 52] == 1


def test_erase_reaches_base_mask_pixels_with_no_live_clicks(ctrl):
    """The motivating use case: a stray leftover pixel from Cutie/SAM2 in
    the *base* mask, with no live click to override it -- erase must
    still reach it (force-to-background, not "revert to nothing")."""
    base = np.zeros((80, 80), dtype=np.uint8)
    base[40, 40] = 2  # a stray mislabeled pixel, no live clicks involved
    ctrl.set_base_mask(base)
    assert ctrl.get_mask()[40, 40] == 2

    mask = ctrl.erase_circle(40, 40, radius=3)
    assert mask[40, 40] == 0


def test_paint_overlay_survives_a_later_click_for_a_different_person(ctrl):
    """The core bug this overlay exists to prevent: _run_predictions()
    rebuilds `combined` from base_mask on every call -- a naive edit to
    self._mask would be wiped by the next click anywhere."""
    ctrl.paint_circle(1, 20, 20, radius=5)
    assert ctrl.get_mask()[20, 20] == 1

    ctrl.push_point(2, 60, 60, positive=True)  # unrelated click, different person

    assert ctrl.get_mask()[20, 20] == 1  # still there


def test_paint_overlay_wins_over_a_later_click_for_the_same_person(ctrl):
    """Documented consequence: once hand-edited, a later SAM2 click for
    the *same* person doesn't override the painted pixels -- the overlay
    always wins until explicitly cleared."""
    ctrl.paint_circle(1, 20, 20, radius=5)
    ctrl.push_point(1, 60, 60, positive=True)  # same person, elsewhere
    assert ctrl.get_mask()[20, 20] == 1


def test_clear_person_also_clears_that_persons_paint_overlay(ctrl):
    ctrl.paint_circle(1, 20, 20, radius=5)
    ctrl.paint_circle(2, 60, 60, radius=5)

    mask = ctrl.clear_person(1)
    assert mask[20, 20] == 0        # person 1's manual edit is gone
    assert mask[60, 60] == 2        # person 2's is untouched


def test_clear_all_resets_paint_overlay(ctrl):
    ctrl.paint_circle(1, 20, 20, radius=5)
    ctrl.clear_all()
    assert ctrl.get_mask()[20, 20] == 0
    # And painting again afterwards still works (overlay lazily rebuilt).
    mask = ctrl.paint_circle(1, 20, 20, radius=5)
    assert mask[20, 20] == 1


def test_clear_paint_overlay_reverts_to_base_and_clicks_only(ctrl):
    base = np.zeros((80, 80), dtype=np.uint8)
    base[10, 10] = 3
    ctrl.set_base_mask(base)
    ctrl.erase_circle(10, 10, radius=2)   # manually erase that base pixel
    assert ctrl.get_mask()[10, 10] == 0

    mask = ctrl.clear_paint_overlay()
    assert mask[10, 10] == 3  # base pixel reappears, manual edit discarded


def test_set_image_resets_paint_overlay_for_the_new_frame():
    ctrl = ClickController(model_name="nonexistent/model-for-tests")
    frame1 = np.zeros((80, 80, 3), dtype=np.uint8)
    ctrl.set_image(frame1)
    ctrl.paint_circle(1, 20, 20, radius=5)
    assert ctrl.get_mask()[20, 20] == 1

    frame2 = np.zeros((80, 80, 3), dtype=np.uint8)
    ctrl.set_image(frame2)
    assert ctrl.get_mask()[20, 20] == 0  # fresh frame, no leftover paint


# ---------------------------------------------------------------------------
# Hydra config clobbering (real bug: SAM2 fails to load on the *second*
# ClickController built in a process, after a Cutie tracking job has run
# in between)
# ---------------------------------------------------------------------------


def test_reinit_sam2_hydra_config_recovers_from_a_clobbered_global_hydra():
    """sam2/__init__.py registers SAM2's own Hydra config search path
    exactly once, guarded by `if not GlobalHydra.instance().is_initialized()`.
    CutieWorker._load_cutie() later calls GlobalHydra.instance().clear()
    and re-initialises Hydra pointed at Cutie's own config dir -- once
    that's happened, sam2's one-time guard never fires again (it's
    top-level module code that already ran at first import), so a later
    ClickController()'s SAM2 build fails with "Cannot find primary
    config 'configs/sam2.1/...yaml'". Confirmed against a real user
    report with this exact error message.

    Reproduces the clobbering directly (not via a real Cutie model load,
    which would need real weights/config on disk) and confirms
    _reinit_sam2_hydra_config() recovers regardless of what Hydra's
    global state held before it ran.
    """
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    from app.pose.cutie_click_controller import _reinit_sam2_hydra_config

    # Simulate Cutie having left Hydra initialized with an unrelated
    # config search path (standing in for Cutie's own config dir).
    GlobalHydra.instance().clear()
    initialize_config_module("hydra.conf", version_base="1.2")

    # Without the fix, this is exactly the reported failure:
    # MissingConfigException: Cannot find primary config
    # 'configs/sam2.1/sam2.1_hiera_b+.yaml'.
    _reinit_sam2_hydra_config()
    cfg = compose(config_name="configs/sam2.1/sam2.1_hiera_b+.yaml")
    assert cfg.model._target_ == "sam2.modeling.sam2_base.SAM2Base"
