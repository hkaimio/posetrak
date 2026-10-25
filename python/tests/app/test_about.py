# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""The About box carries the notices that the licences of the SOMA-77 pose model ask for."""
from app.ui.about import about_html


def test_about_box_has_the_nvidia_notice_and_built_with_dinov3():
    text = about_html()
    assert "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" in text
    assert "Built with DINOv3" in text
