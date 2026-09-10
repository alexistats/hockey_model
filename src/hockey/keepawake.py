"""Stop Windows suspending the machine while a long fit runs.

A full-pool fit runs for hours with no keyboard or mouse activity, which is
exactly what the idle timer is watching for. Two overnight runs were lost this
way, and the failure is easy to misread: the process simply stops, and whatever
notices it reports the symptom rather than the cause.

SetThreadExecutionState is the documented way to say "keep the system awake,
I'm working". It needs no elevation, affects only this process, and is undone
when the flag is cleared or the process exits, so a crashed run cannot leave
the machine permanently awake.

It does NOT keep the display on, and it does not override a lid close or an
explicit sleep from the Start menu. Those are deliberate acts; this only
prevents the idle timeout.
"""

import contextlib
import ctypes
import logging
import sys

logger = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
# Lets the machine keep working with the display off on modern-standby
# hardware; harmless where it is not supported.
ES_AWAYMODE_REQUIRED = 0x00000040


def _set(flags: int) -> bool:
    if not sys.platform.startswith("win"):
        return False
    result = ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
    return result != 0


@contextlib.contextmanager
def keep_awake(reason: str = "long-running fit"):
    """Hold the system awake for the duration of the block."""
    held = _set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
    if not held:
        # Away mode is not available everywhere; the plain request usually is.
        held = _set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if held:
        logger.info("holding the system awake (%s)", reason)
    else:
        logger.warning(
            "could not request that the system stay awake; a long run may be "
            "cut short by the idle timeout"
        )
    try:
        yield held
    finally:
        if held:
            _set(ES_CONTINUOUS)
            logger.info("released the wake request")
