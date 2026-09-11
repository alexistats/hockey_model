"""Progress reporting for runs that take hours.

A staged fit spends most of its life inside a sampler that prints nothing, and
its output is usually redirected to a log, so "is it still working and how far
along" has been unanswerable without reading timestamps and doing arithmetic.

Two channels, because they answer different questions. A line in the log after
each step, for reading afterwards; and a single-line file overwritten in place,
for glancing at while it runs:

    type artifacts\\board_full\\progress.txt

The estimate is the mean of the steps finished so far rather than the last one,
since batch times vary and the last batch is a poor predictor on its own. It is
honest about being an estimate: early on, with one step done, it says so.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@dataclass
class Progress:
    """Step-level progress with an estimate of what is left."""

    total: int
    label: str = "step"
    path: Path | None = None
    started: float = field(default_factory=time.monotonic)
    step_times: list[float] = field(default_factory=list)
    _last_mark: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self._write(f"{self.label}: starting, {self.total} to do")

    @property
    def done(self) -> int:
        return len(self.step_times)

    def _write(self, line: str) -> None:
        logger.info("%s", line)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Overwritten rather than appended: this file answers "where is it
            # now", and a reader should not have to scroll to find out.
            self.path.write_text(line + "\n", encoding="utf-8")

    def advance(self, note: str = "") -> str:
        """Record one completed step and report."""
        now = time.monotonic()
        self.step_times.append(now - self._last_mark)
        self._last_mark = now

        elapsed = now - self.started
        mean_step = sum(self.step_times) / len(self.step_times)
        remaining = mean_step * (self.total - self.done)
        pct = 100.0 * self.done / self.total

        estimate = "estimating" if self.done < 2 else f"about {format_duration(remaining)} left"
        line = (
            f"{self.label} {self.done}/{self.total} ({pct:.0f}%)  "
            f"elapsed {format_duration(elapsed)}  "
            f"last {format_duration(self.step_times[-1])}  {estimate}"
        )
        if note:
            line += f"  |  {note}"
        self._write(line)
        return line

    def finish(self) -> str:
        elapsed = time.monotonic() - self.started
        line = f"{self.label}: all {self.total} done in {format_duration(elapsed)}"
        self._write(line)
        return line
