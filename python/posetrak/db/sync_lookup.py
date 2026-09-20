# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""sync_lookup.py — frame-to-time lookup for one sync configuration."""
from __future__ import annotations

import sqlite3


def load_sync_table(session: sqlite3.Connection, sync_config_id: str):
    """Build the ``SyncTable`` that maps a video's frame numbers to global time.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    sync_config_id:
        The sync configuration whose points to load.

    Returns
    -------
    app.setup.db_context.SyncTable
        ``frame_to_global_time(frame, shot_video_id)`` returns ``None`` for a
        video the configuration has no points for.
    """
    # Imported here: it lives in the GUI application's package.
    from app.setup.db_context import SyncPoint, SyncTable

    rows = session.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
        "FROM sync_points sp JOIN capture_videos sv ON sv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id = ?",
        (sync_config_id,),
    ).fetchall()
    return SyncTable(
        [
            SyncPoint(camera_instance_id="", shot_video_id=r[0], video_frame=r[1], timestamp_s=r[2])
            for r in rows
        ],
        {r[0]: float(r[3]) for r in rows},
    )
