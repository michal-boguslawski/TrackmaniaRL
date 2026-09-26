"""Compatibility imports for shared inference APIs; implementation lives in ``evaluation``."""

from rl_lib.evaluation.inference import EpisodeResult, log_evaluation_results, run_inference

__all__ = ["EpisodeResult", "log_evaluation_results", "run_inference"]
