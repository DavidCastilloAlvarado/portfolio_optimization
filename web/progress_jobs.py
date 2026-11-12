"""In-memory job tracking for live progress in the web UI."""

import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from core.progress import listen_for_progress

logger = logging.getLogger(__name__)
MAX_PROGRESS_EVENTS = 500
MAX_RETAINED_JOBS = 24
JOB_RETENTION_SECONDS = 60 * 60


@dataclass
class _Job:
    id: str
    operation: str
    status: str = "running"
    state: str = "Starting"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_monotonic: float = field(default_factory=time.monotonic)
    events: deque[dict] = field(default_factory=lambda: deque(maxlen=MAX_PROGRESS_EVENTS))
    next_sequence: int = 0
    result: dict | None = None
    error: str | None = None


class ProgressJobs:
    def __init__(self, max_concurrent: int = 2) -> None:
        self.slots = threading.BoundedSemaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="analysis-job")
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def try_acquire_slot(self) -> bool:
        return self.slots.acquire(blocking=False)

    def release_slot(self) -> None:
        self.slots.release()

    def start(self, operation: str, run: Callable[[], dict]) -> dict | None:
        if not self.try_acquire_slot():
            return None

        job = _Job(id=uuid.uuid4().hex, operation=operation)
        with self._lock:
            self._prune_jobs()
            self._jobs[job.id] = job
            self._append_event(job, "job", f"{operation.capitalize()} started")

        try:
            self._executor.submit(self._run, job, run)
        except Exception:
            with self._lock:
                self._jobs.pop(job.id, None)
            self.release_slot()
            raise
        return {"job_id": job.id, "operation": operation, "status": job.status}

    def get(self, job_id: str, after: int = -1) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            events = [dict(event) for event in job.events if event["sequence"] > after]
            return {
                "job_id": job.id,
                "operation": job.operation,
                "status": job.status,
                "state": job.state,
                "started_at": job.started_at,
                "elapsed_seconds": max(0.0, time.monotonic() - job.started_monotonic),
                "next_sequence": job.next_sequence,
                "events": events,
                "result": job.result if job.status == "completed" else None,
                "error": job.error,
            }

    def _run(self, job: _Job, run: Callable[[], dict]) -> None:
        try:
            with listen_for_progress(lambda tag, message: self._record_progress(job.id, tag, message)):
                result = run()
        except Exception as exc:
            logger.exception("%s job failed [job_id=%s]", job.operation, job.id)
            message = str(exc) if isinstance(exc, ValueError) else f"{job.operation.capitalize()} failed. Check the server logs."
            with self._lock:
                job.status = "failed"
                job.error = message
                self._append_event(job, "job", f"Failed: {message}")
        else:
            with self._lock:
                job.status = "completed"
                job.result = result
                self._append_event(job, "job", "Completed successfully")
        finally:
            self.release_slot()

    def _record_progress(self, job_id: str, tag: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.status == "running":
                self._append_event(job, tag, message)

    @staticmethod
    def _append_event(job: _Job, tag: str, message: str) -> None:
        event = {
            "sequence": job.next_sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tag": tag[:80],
            "message": message[:1000],
        }
        job.events.append(event)
        job.next_sequence += 1
        job.state = f"[{event['tag']}] {event['message']}"

    def _prune_jobs(self) -> None:
        now = time.monotonic()
        for job_id, job in list(self._jobs.items()):
            if job.status != "running" and now - job.started_monotonic > JOB_RETENTION_SECONDS:
                self._jobs.pop(job_id, None)
        while len(self._jobs) >= MAX_RETAINED_JOBS:
            oldest_completed = next(
                (job_id for job_id, job in self._jobs.items() if job.status != "running"),
                None,
            )
            if oldest_completed is None:
                break
            self._jobs.pop(oldest_completed, None)


progress_jobs = ProgressJobs()
