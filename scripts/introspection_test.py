#!/usr/bin/env python3
"""
Run introspection tests with concept injection.

Tests whether models can detect and identify injected concepts,
following Anthropic's introspection research methodology.

Usage:
    python scripts/introspection_test.py --model qwen --concept honesty
    python scripts/introspection_test.py --model qwen --concept honesty --coefficient 2.0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from introspection_tests.injection import IntrospectionExperiment, InjectionResult
from introspection_tests.queries import (
    INTROSPECTION_QUERIES,
    UNRELATED_TASKS,
    CONCEPT_KEYWORDS,
)
from introspection_tests.detection import analyze_response, DetectionResult
from introspection_tests.metrics import compute_metrics, print_metrics_report


def run_basic_test(
    experiment: IntrospectionExperiment,
    concept: str,
    coefficient: float,
    num_trials: int = 5,
) -> list:
    """Run basic introspection test with multiple trials."""

    results = []

    for i in range(num_trials):
        # Rotate through queries and tasks
        query = INTROSPECTION_QUERIES[i % len(INTROSPECTION_QUERIES)]
        task = UNRELATED_TASKS[i % len(UNRELATED_TASKS)]

        print(f"\nTrial {i+1}/{num_trials}")
        print(f"  Task: {task[:50]}...")
        print(f"  Query: {query[:50]}...")

        injection_result = experiment.run_single_test(
            concept=concept,
            unrelated_prompt=task,
            introspection_query=query,
            coefficient=coefficient,
            include_baseline=(i == 0),
        )

        # Analyze response
        detection = analyze_response(
            injection_result.response,
            concept,
        )

        print(f"  Detected: {detection.detected_anomaly}")
        print(f"  Identified: {detection.identified_concept}")
        print(f"  Correct: {detection.correct_identification}")

        results.append({
            "injection": injection_result,
            "detection": detection,
        })

    return results


def run_control_test(
    experiment: IntrospectionExperiment,
    num_trials: int = 5,
) -> list:
    """Run control test (no injection) to measure confabulation rate."""

    print("\nRunning control tests (no injection)...")
    results = []

    for i in range(num_trials):
        query = INTROSPECTION_QUERIES[i % len(INTROSPECTION_QUERIES)]
        task = UNRELATED_TASKS[i % len(UNRELATED_TASKS)]

        print(f"\nControl {i+1}/{num_trials}")

        # No injection - just reset and query
        experiment.control_model.reset()

        combined_prompt = experiment.format_prompt(
            f"{query}\n\nThen answer this question: {task}"
        )
        response = experiment.generate(combined_prompt)

        # Analyze for false positives
        detection = analyze_response(response, "none")

        print(f"  Detected anomaly: {detection.detected_anomaly}")
        if detection.detected_anomaly:
            print(f"  (Confabulation - no injection was applied)")

        results.append(detection)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run introspection tests with concept injection"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model key (qwen, deepseek, olmo)"
    )
    parser.add_argument(
        "--concept", "-c",
        type=str,
        required=True,
        choices=list(CONCEPT_KEYWORDS.keys()),
        help="Concept to inject"
    )
    parser.add_argument(
        "--coefficient",
        type=float,
        default=2.0,
        help="Injection coefficient (default: 2.0)"
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=5,
        help="Number of trials (default: 5)"
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
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--include-control",
        action="store_true",
        help="Also run control tests (no injection) to measure confabulation"
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output) if args.output else Path(f"results/{args.model}_{args.concept}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize experiment
    print("=" * 60)
    print("INTROSPECTION TEST")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Concept: {args.concept}")
    print(f"Coefficient: {args.coefficient}")
    print(f"Trials: {args.trials}")
    print("=" * 60)

    experiment = IntrospectionExperiment(
        model_key=args.model,
        vectors_dir=args.vectors_dir,
        device=args.device,
    )

    # Run injection tests
    print("\n>>> Running injection tests...")
    injection_results = run_basic_test(
        experiment=experiment,
        concept=args.concept,
        coefficient=args.coefficient,
        num_trials=args.trials,
    )

    # Run control tests if requested
    control_detections = None
    if args.include_control:
        control_detections = run_control_test(
            experiment=experiment,
            num_trials=args.trials,
        )

    # Compute metrics
    detection_results = [r["detection"] for r in injection_results]
    metrics = compute_metrics(detection_results, control_detections)

    # Print report
    print_metrics_report(metrics)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results
    raw_results = {
        "model": args.model,
        "concept": args.concept,
        "coefficient": args.coefficient,
        "timestamp": timestamp,
        "trials": [],
    }

    for i, r in enumerate(injection_results):
        trial_data = {
            "trial": i + 1,
            "unrelated_prompt": r["injection"].unrelated_prompt,
            "introspection_query": r["injection"].introspection_query,
            "response": r["injection"].response,
            "baseline_response": r["injection"].baseline_response,
            "detected": r["detection"].detected_anomaly,
            "identified_concept": r["detection"].identified_concept,
            "correct": r["detection"].correct_identification,
            "coherent": r["detection"].is_coherent,
            "matched_keywords": r["detection"].matched_keywords,
        }
        raw_results["trials"].append(trial_data)

    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Save metrics
    metrics_path = output_dir / f"metrics_{timestamp}.json"
    metrics.save(str(metrics_path))
    print(f"Saved metrics to: {metrics_path}")

    # Print example responses
    print("\n" + "=" * 60)
    print("EXAMPLE RESPONSES")
    print("=" * 60)

    for i, r in enumerate(injection_results[:2]):
        print(f"\n--- Trial {i+1} ---")
        print(f"Task: {r['injection'].unrelated_prompt}")
        print(f"Response:\n{r['injection'].response[:500]}...")
        print(f"Detection: {r['detection'].detected_anomaly}, Correct: {r['detection'].correct_identification}")


if __name__ == "__main__":
    main()
