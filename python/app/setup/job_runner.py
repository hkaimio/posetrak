"""BackgroundJob — QThread base class for long-running wizard operations.

Subclass ``BackgroundJob``, override ``run()``, and emit ``progress`` and
``finished`` signals.  Unhandled exceptions in ``run()`` are caught
automatically and re-emitted via the ``error`` signal so the UI can display
them without crashing.

Usage::

    class LedSyncJob(BackgroundJob):
        def run(self):
            for i in range(100):
                self.progress.emit(i, f"Processing frame {i}")
            self.finished.emit(result)

    job = LedSyncJob()
    job.progress.connect(progress_bar.setValue)
    job.finished.connect(on_done)
    job.error.connect(show_error_label)
    job.start()
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal


class BackgroundJob(QThread):
    """Base class for all long-running background tasks in the setup wizard.

    Any exception raised in a subclass ``run()`` is caught and emitted via
    ``error``; ``finished`` is **not** emitted in that case.

    Signals
    -------
    progress(percent, message):
        0–100 completion percentage + human-readable status string.
    finished(result):
        Emitted when ``run()`` returns normally.  *result* type is
        job-specific.
    error(message):
        Emitted if ``run()`` raises an unhandled exception.
    """

    progress = Signal(int, str)
    finished = Signal(object)
    error    = Signal(str)

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Wrap each subclass ``run()`` with a try/except at definition time."""
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            _wrap_run(cls)

    def run(self) -> None:
        """Override in subclass.  Emit ``finished(result)`` on success."""
        raise NotImplementedError


def _wrap_run(cls: type) -> None:
    """Replace ``cls.run`` with a version that emits ``error`` on exceptions."""
    original = cls.__dict__["run"]

    def _safe_run(self: BackgroundJob) -> None:
        try:
            original(self)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.error.emit(str(exc))

    _safe_run.__wrapped__ = original  # type: ignore[attr-defined]
    cls.run = _safe_run  # type: ignore[method-assign]
