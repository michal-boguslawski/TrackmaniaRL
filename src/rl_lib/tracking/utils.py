"""Utilities shared by tracking backends and run lifecycle management."""


def run_status_for_exception(
    exc_type: type[BaseException] | None, exc_val: BaseException | None = None
) -> str:
    """Map the exception leaving a run's context onto its final status.

    Interruptions are marked ``KILLED`` rather than ``FAILED``. A
    ``SystemExit(0)`` is a deliberate clean exit and remains ``FINISHED``.

    Args:
        exc_type: Exception type from ``__exit__``, or ``None`` on success.
        exc_val: Exception value, if one was raised.

    Returns:
        Run status string: ``FINISHED``, ``KILLED``, or ``FAILED``.
    """
    if exc_type is None:
        return "FINISHED"
    if issubclass(exc_type, KeyboardInterrupt):
        return "KILLED"
    if (
        issubclass(exc_type, SystemExit)
        and exc_val is not None
        and getattr(exc_val, "code", None) in (0, None)
    ):
        return "FINISHED"
    return "FAILED"
