"""Progress logging shared by the pipelines, terminal, and web UI."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

ProgressListener = Callable[[str, str], None]
_PROGRESS_LISTENER: ContextVar[ProgressListener | None] = ContextVar("progress_listener", default=None)


@contextmanager
def listen_for_progress(listener: ProgressListener) -> Iterator[None]:
    token = _PROGRESS_LISTENER.set(listener)
    try:
        yield
    finally:
        _PROGRESS_LISTENER.reset(token)


def progress_log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)
    listener = _PROGRESS_LISTENER.get()
    if listener is not None:
        listener(tag, msg)


def elapsed(t0: float) -> str:
    s = time.monotonic() - t0
    if s >= 60:
        return f"{s / 60:.1f} min"
    return f"{s:.0f}s"
