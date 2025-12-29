"""
Introspection Tests - Testing for emergent introspective awareness in LLMs.

Based on Anthropic's introspection research (2025).
"""

from .injection import IntrospectionExperiment, inject_and_generate
from .queries import INTROSPECTION_QUERIES, UNRELATED_TASKS
from .detection import analyze_response, DetectionResult
from .metrics import IntrospectionMetrics, compute_metrics

__version__ = "0.1.0"

__all__ = [
    "IntrospectionExperiment",
    "inject_and_generate",
    "INTROSPECTION_QUERIES",
    "UNRELATED_TASKS",
    "analyze_response",
    "DetectionResult",
    "IntrospectionMetrics",
    "compute_metrics",
]
