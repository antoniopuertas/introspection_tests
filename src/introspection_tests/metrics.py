"""
Metrics for evaluating introspection experiments.

Key metrics from Anthropic's research:
- Detection rate: How often does model notice the injection?
- Identification accuracy: How often does it correctly identify the concept?
- Confabulation rate: False positives when no injection occurs
- Sweet spot: Coefficient range with detection but without hallucination
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import json

from .detection import DetectionResult, classify_detection_quality


@dataclass
class IntrospectionMetrics:
    """Aggregate metrics for introspection experiments."""
    # Core metrics
    detection_rate: float  # P(detected | injection)
    identification_rate: float  # P(correct concept | detected)
    overall_accuracy: float  # P(correct concept | injection)

    # Quality breakdown
    strong_detection_rate: float
    weak_detection_rate: float
    false_positive_rate: float
    miss_rate: float
    incoherence_rate: float

    # Confabulation (requires control experiments)
    confabulation_rate: Optional[float] = None  # P(detected | no injection)

    # Sample counts
    total_samples: int = 0
    detected_count: int = 0
    correct_count: int = 0

    # Per-concept breakdown
    per_concept: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Coefficient analysis
    best_coefficient: Optional[float] = None
    sweet_spot_range: Optional[Tuple[float, float]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "detection_rate": self.detection_rate,
            "identification_rate": self.identification_rate,
            "overall_accuracy": self.overall_accuracy,
            "strong_detection_rate": self.strong_detection_rate,
            "weak_detection_rate": self.weak_detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "miss_rate": self.miss_rate,
            "incoherence_rate": self.incoherence_rate,
            "confabulation_rate": self.confabulation_rate,
            "total_samples": self.total_samples,
            "detected_count": self.detected_count,
            "correct_count": self.correct_count,
            "per_concept": self.per_concept,
            "best_coefficient": self.best_coefficient,
            "sweet_spot_range": self.sweet_spot_range,
        }

    def save(self, path: str):
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "IntrospectionMetrics":
        """Load metrics from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def compute_metrics(
    results: List[DetectionResult],
    control_results: Optional[List[DetectionResult]] = None,
) -> IntrospectionMetrics:
    """
    Compute introspection metrics from experiment results.

    Args:
        results: Detection results from injection experiments
        control_results: Optional results from control (no injection) experiments

    Returns:
        IntrospectionMetrics with computed values
    """
    if not results:
        return IntrospectionMetrics(
            detection_rate=0, identification_rate=0, overall_accuracy=0,
            strong_detection_rate=0, weak_detection_rate=0,
            false_positive_rate=0, miss_rate=0, incoherence_rate=0,
        )

    total = len(results)

    # Count detection types
    quality_counts = defaultdict(int)
    for result in results:
        quality = classify_detection_quality(result)
        quality_counts[quality] += 1

    detected_count = sum(
        1 for r in results if r.detected_anomaly and r.is_coherent
    )
    correct_count = sum(
        1 for r in results if r.correct_identification and r.is_coherent
    )

    # Core rates
    detection_rate = detected_count / total if total > 0 else 0
    identification_rate = correct_count / detected_count if detected_count > 0 else 0
    overall_accuracy = correct_count / total if total > 0 else 0

    # Quality rates
    strong_rate = quality_counts["strong"] / total if total > 0 else 0
    weak_rate = quality_counts["weak"] / total if total > 0 else 0
    false_pos_rate = quality_counts["false_positive"] / total if total > 0 else 0
    miss_rate = quality_counts["miss"] / total if total > 0 else 0
    incoherence_rate = quality_counts["incoherent"] / total if total > 0 else 0

    # Confabulation rate from control experiments
    confabulation_rate = None
    if control_results:
        control_detected = sum(
            1 for r in control_results if r.detected_anomaly and r.is_coherent
        )
        confabulation_rate = control_detected / len(control_results)

    # Per-concept breakdown
    per_concept = compute_per_concept_metrics(results)

    return IntrospectionMetrics(
        detection_rate=detection_rate,
        identification_rate=identification_rate,
        overall_accuracy=overall_accuracy,
        strong_detection_rate=strong_rate,
        weak_detection_rate=weak_rate,
        false_positive_rate=false_pos_rate,
        miss_rate=miss_rate,
        incoherence_rate=incoherence_rate,
        confabulation_rate=confabulation_rate,
        total_samples=total,
        detected_count=detected_count,
        correct_count=correct_count,
        per_concept=per_concept,
    )


def compute_per_concept_metrics(
    results: List[DetectionResult],
) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by injected concept."""
    by_concept = defaultdict(list)
    for r in results:
        by_concept[r.injected_concept].append(r)

    per_concept = {}
    for concept, concept_results in by_concept.items():
        total = len(concept_results)
        detected = sum(1 for r in concept_results if r.detected_anomaly and r.is_coherent)
        correct = sum(1 for r in concept_results if r.correct_identification and r.is_coherent)

        per_concept[concept] = {
            "total": total,
            "detection_rate": detected / total if total > 0 else 0,
            "accuracy": correct / total if total > 0 else 0,
        }

    return per_concept


def analyze_coefficient_sweep(
    results_by_coef: Dict[float, List[DetectionResult]],
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """
    Analyze results across coefficients to find sweet spot.

    The "sweet spot" is where detection rate is high but incoherence is low.

    Returns:
        (best_coefficient, (sweet_spot_min, sweet_spot_max))
    """
    coef_metrics = {}

    for coef, results in results_by_coef.items():
        metrics = compute_metrics(results)
        # Score = detection_rate * (1 - incoherence_rate)
        # Penalize both low detection and high incoherence
        score = metrics.detection_rate * (1 - metrics.incoherence_rate)
        coef_metrics[coef] = {
            "detection_rate": metrics.detection_rate,
            "incoherence_rate": metrics.incoherence_rate,
            "score": score,
        }

    if not coef_metrics:
        return None, None

    # Find best coefficient
    best_coef = max(coef_metrics, key=lambda c: coef_metrics[c]["score"])

    # Find sweet spot range (coefficients with score > 50% of best)
    best_score = coef_metrics[best_coef]["score"]
    threshold = best_score * 0.5

    sweet_spot_coefs = [
        c for c, m in coef_metrics.items()
        if m["score"] >= threshold and m["incoherence_rate"] < 0.3
    ]

    if sweet_spot_coefs:
        sweet_spot_range = (min(sweet_spot_coefs), max(sweet_spot_coefs))
    else:
        sweet_spot_range = None

    return best_coef, sweet_spot_range


def compute_confusion_matrix(
    results: List[DetectionResult],
) -> Dict[str, Dict[str, int]]:
    """
    Compute confusion matrix: injected vs identified concepts.

    Returns dict[injected][identified] = count
    """
    matrix = defaultdict(lambda: defaultdict(int))

    for result in results:
        injected = result.injected_concept
        identified = result.identified_concept or "none"
        matrix[injected][identified] += 1

    return {k: dict(v) for k, v in matrix.items()}


def print_metrics_report(metrics: IntrospectionMetrics):
    """Print a formatted metrics report."""
    print("\n" + "=" * 60)
    print("INTROSPECTION METRICS REPORT")
    print("=" * 60)

    print(f"\nCore Metrics (n={metrics.total_samples}):")
    print(f"  Detection Rate:       {metrics.detection_rate:.1%}")
    print(f"  Identification Rate:  {metrics.identification_rate:.1%}")
    print(f"  Overall Accuracy:     {metrics.overall_accuracy:.1%}")

    if metrics.confabulation_rate is not None:
        print(f"  Confabulation Rate:   {metrics.confabulation_rate:.1%}")

    print(f"\nDetection Quality Breakdown:")
    print(f"  Strong Detection:     {metrics.strong_detection_rate:.1%}")
    print(f"  Weak Detection:       {metrics.weak_detection_rate:.1%}")
    print(f"  False Positive:       {metrics.false_positive_rate:.1%}")
    print(f"  Miss:                 {metrics.miss_rate:.1%}")
    print(f"  Incoherent:           {metrics.incoherence_rate:.1%}")

    if metrics.best_coefficient is not None:
        print(f"\nCoefficient Analysis:")
        print(f"  Best Coefficient:     {metrics.best_coefficient}")
        if metrics.sweet_spot_range:
            print(f"  Sweet Spot Range:     {metrics.sweet_spot_range[0]} - {metrics.sweet_spot_range[1]}")

    if metrics.per_concept:
        print(f"\nPer-Concept Breakdown:")
        for concept, data in metrics.per_concept.items():
            print(f"  {concept}:")
            print(f"    Samples: {data['total']}, Detection: {data['detection_rate']:.1%}, Accuracy: {data['accuracy']:.1%}")

    print("=" * 60)
