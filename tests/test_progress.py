"""Progress reporting, including that its estimate is honest early on."""

import time

from hockey.progress import Progress, format_duration


def test_duration_formatting():
    assert format_duration(45) == "45s"
    assert format_duration(125) == "2m05s"
    assert format_duration(3725) == "1h02m"
    assert format_duration(-5) == "0s"


def test_progress_writes_a_single_line_file(tmp_path):
    """The file answers 'where is it now', so it is overwritten rather than
    appended - a reader should not have to scroll to find the current state."""
    path = tmp_path / "progress.txt"
    progress = Progress(total=3, label="batch", path=path)
    progress.advance()
    progress.advance()
    assert len(path.read_text().strip().splitlines()) == 1
    assert "batch 2/3" in path.read_text()


def test_no_estimate_until_there_is_something_to_estimate_from(tmp_path):
    progress = Progress(total=4, label="batch", path=tmp_path / "p.txt")
    assert "estimating" in progress.advance()
    assert "left" in progress.advance()


def test_estimate_uses_the_mean_not_the_last_step(tmp_path):
    """Batch times vary, so the most recent one is a poor predictor alone."""
    progress = Progress(total=10, label="batch", path=tmp_path / "p.txt")
    progress.step_times = [10.0, 10.0, 40.0]
    progress._last_mark = time.monotonic()
    line = progress.advance()
    # Four steps averaging about 15s each, six left, so roughly 90s - not the
    # 0s a last-step estimate would give after a fast final step.
    assert "left" in line
    assert progress.done == 4


def test_percentage_and_counts(tmp_path):
    progress = Progress(total=4, label="batch", path=tmp_path / "p.txt")
    progress.advance()
    line = progress.advance()
    assert "2/4" in line
    assert "50%" in line


def test_finish_reports_the_total(tmp_path):
    path = tmp_path / "p.txt"
    progress = Progress(total=2, label="batch", path=path)
    progress.advance()
    progress.advance()
    line = progress.finish()
    assert "all 2 done" in line
    assert "all 2 done" in path.read_text()


def test_works_without_a_file():
    progress = Progress(total=2, label="batch")
    assert "1/2" in progress.advance()
