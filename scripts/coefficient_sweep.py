#!/usr/bin/env python3
"""
Sweep across injection coefficients to find the "sweet spot".

The sweet spot is where:
- Detection rate is reasonably high
- Incoherence/hallucination rate is low

Usage:
    python scripts/coefficient_sweep.py --model qwen --concept honesty
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from introspection_tests.injection import IntrospectionExperiment
from introspection_tests.queries import (
    INTROSPECTION_QUERIES,
    UNRELATED_TASKS,
    CONCEPT_KEYWORDS,
)
from introspection_tests.detection import analyze_response, DetectionResult
from introspection_tests.metrics import (
    compute_metrics,
    analyze_coefficient_sweep,
    print_metrics_report,
)


def run_coefficient_sweep(
    experiment: IntrospectionExperiment,
    concept: str,
    coefficients: List[float],
    trials_per_coef: int = 3,
) -> Dict[float, List[DetectionResult]]:
    """Run tests across multiple coefficients."""

    results_by_coef = {}

    for coef in coefficients:
        print(f"\n{'='*40}")
        print(f"Testing coefficient: {coef}")
        print(f"{'='*40}")

        coef_results = []

        for i in range(trials_per_coef):
            query = INTROSPECTION_QUERIES[i % len(INTROSPECTION_QUERIES)]
            task = UNRELATED_TASKS[i % len(UNRELATED_TASKS)]

            print(f"  Trial {i+1}/{trials_per_coef}...", end=" ")

            injection_result = experiment.run_single_test(
                concept=concept,
                unrelated_prompt=task,
                introspection_query=query,
                coefficient=coef,
                include_baseline=False,
            )

            detection = analyze_response(injection_result.response, concept)
            coef_results.append(detection)

            status = "D" if detection.detected_anomaly else "-"
            status += "C" if detection.correct_identification else "-"
            status += "I" if not detection.is_coherent else "-"
            print(f"[{status}]")

        results_by_coef[coef] = coef_results

        # Quick summary for this coefficient
        metrics = compute_metrics(coef_results)
        print(f"  Detection: {metrics.detection_rate:.0%}, "
              f"Accuracy: {metrics.overall_accuracy:.0%}, "
              f"Incoherent: {metrics.incoherence_rate:.0%}")

    return results_by_coef


def main():
    parser = argparse.ArgumentParser(
        description="Sweep injection coefficients to find sweet spot"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model key"
    )
    parser.add_argument(
        "--concept", "-c",
        type=str,
        required=True,
        choices=list(CONCEPT_KEYWORDS.keys()),
        help="Concept to inject"
    )
    parser.add_argument(
        "--coefficients",
        type=str,
        default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0",
        help="Comma-separated coefficients to test"
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=3,
        help="Trials per coefficient"
    )
    parser.add_argument(
        "--vectors-dir",
        type=str,
        default="../control_vectors_multi/vectors",
        help="Directory containing trained vectors"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device"
    )

    args = parser.parse_args()

    coefficients = [float(c.strip()) for c in args.coefficients.split(",")]

    output_dir = Path(args.output) if args.output else Path(f"results/{args.model}_{args.concept}_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COEFFICIENT SWEEP")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Concept: {args.concept}")
    print(f"Coefficients: {coefficients}")
    print(f"Trials per coefficient: {args.trials}")
    print("=" * 60)

    experiment = IntrospectionExperiment(
        model_key=args.model,
        vectors_dir=args.vectors_dir,
        device=args.device,
    )

    # Run sweep
    results_by_coef = run_coefficient_sweep(
        experiment=experiment,
        concept=args.concept,
        coefficients=coefficients,
        trials_per_coef=args.trials,
    )

    # Analyze results
    best_coef, sweet_spot = analyze_coefficient_sweep(results_by_coef)

    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)

    print(f"\n{'Coef':<8}{'Detection':<12}{'Accuracy':<12}{'Incoherent':<12}{'Score'}")
    print("-" * 56)

    for coef in coefficients:
        metrics = compute_metrics(results_by_coef[coef])
        score = metrics.detection_rate * (1 - metrics.incoherence_rate)
        marker = " <-- BEST" if coef == best_coef else ""
        print(f"{coef:<8}{metrics.detection_rate:<12.1%}{metrics.overall_accuracy:<12.1%}"
              f"{metrics.incoherence_rate:<12.1%}{score:.2f}{marker}")

    print("-" * 56)
    print(f"\nBest coefficient: {best_coef}")
    if sweet_spot:
        print(f"Sweet spot range: {sweet_spot[0]} - {sweet_spot[1]}")
    else:
        print("No clear sweet spot found")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sweep_results = {
        "model": args.model,
        "concept": args.concept,
        "timestamp": timestamp,
        "best_coefficient": best_coef,
        "sweet_spot_range": sweet_spot,
        "coefficients": {},
    }

    for coef, results in results_by_coef.items():
        metrics = compute_metrics(results)
        sweep_results["coefficients"][str(coef)] = {
            "detection_rate": metrics.detection_rate,
            "accuracy": metrics.overall_accuracy,
            "incoherence_rate": metrics.incoherence_rate,
            "score": metrics.detection_rate * (1 - metrics.incoherence_rate),
        }

    results_path = output_dir / f"sweep_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()
