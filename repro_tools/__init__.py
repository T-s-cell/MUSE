"""Reproduction instrumentation for MUSE ETTh1 phase-1 preflight.

All hooks are disabled unless MUSE_REPRO_ENABLE=1. When disabled, the only
overhead on the official training path is a single None check per call site.
"""

_HOOK = None
_RESOLVED = False


def get_hook():
    global _HOOK, _RESOLVED
    if not _RESOLVED:
        _RESOLVED = True
        import os

        if os.environ.get("MUSE_REPRO_ENABLE", "") == "1":
            from .hook import ReproHook

            _HOOK = ReproHook.from_env()
    return _HOOK
