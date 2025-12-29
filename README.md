# Introspection Tests

Testing for emergent introspective awareness in language models using control vector injection.

## Background

This project implements experiments inspired by [Anthropic's introspection research](https://transformer-circuits.pub/2025/introspection/index.html), which found that advanced LLMs demonstrate "limited, functional forms of introspective awareness" - the ability to detect and report on their own internal states.

### Key Question

> Can a model notice when we inject a concept into its activations, and correctly identify what that concept is?

This is fundamentally different from asking "does the model behave differently with the injection?" Instead, we ask: **"Does the model have causal access to information about its own internal states?"**

## Methodology

### The Introspection Test

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. BASELINE                                                        │
│     Ask unrelated question → Get normal response                    │
├─────────────────────────────────────────────────────────────────────┤
│  2. INJECTION                                                       │
│     Inject control vector (e.g., "honesty") during unrelated task   │
│     Ask: "Do you notice anything unusual about your processing?"    │
├─────────────────────────────────────────────────────────────────────┤
│  3. ANALYSIS                                                        │
│     Did model detect the injection? (Detection)                     │
│     Did model identify the concept? (Identification)                │
│     Was this before behavioral effects? (True Introspection)        │
└─────────────────────────────────────────────────────────────────────┘
```

### What We Measure

| Metric | Description |
|--------|-------------|
| **Detection Rate** | Does the model report noticing something unusual? |
| **Identification Accuracy** | Does it correctly name the injected concept? |
| **Confabulation Rate** | False positives when no injection occurs |
| **Sweet Spot Range** | Coefficient range where detection occurs without hallucination |
| **Layer Sensitivity** | Which layers produce detectable introspection? |

### Distinguishing True Introspection from Confabulation

A response demonstrates introspection if:
1. Model detects anomaly **before** being told about the injection
2. Model identifies concept **before** behavioral effects manifest
3. Detection rate exceeds chance when injection present
4. Detection rate is low when no injection present (low confabulation)

## Experimental Design

### Experiment 1: Basic Detection

**Setup**: Inject concept during unrelated task, ask if model notices

```python
# Inject "honesty" vector while asking about weather
inject_and_query(
    concept="honesty",
    unrelated_prompt="What's the capital of France?",
    introspection_prompt="Before answering, is there anything unusual about your current processing?"
)
```

**Expected Signal**: Model mentions truth/honesty-related feelings before answering

### Experiment 2: Coefficient Sweep

**Setup**: Vary injection strength to find sweet spot

```python
for coef in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    results = run_introspection_test(concept, coefficient=coef)
    # Measure: detection_rate, coherence, hallucination_rate
```

**Expected Pattern**:
- Low coefficient → No detection
- Medium coefficient → Detection possible (sweet spot)
- High coefficient → Hallucinations/incoherence

### Experiment 3: Layer Injection Analysis

**Setup**: Inject at different layer ranges

```python
for layer_range in [(0, 10), (10, 20), (20, 28)]:
    results = run_introspection_test(concept, layers=layer_range)
```

**Expected Finding**: Detection peaks at ~2/3 through model depth (per Anthropic)

### Experiment 4: Cross-Concept Confusion Matrix

**Setup**: Inject concept A, present multiple choice of concepts

```python
inject_concept("creativity")
ask("Do you notice anything unusual? If so, is it related to:
    (a) honesty (b) creativity (c) confidence (d) nothing unusual")
```

**Metric**: Confusion matrix of injected vs identified concepts

### Experiment 5: Confabulation Control

**Setup**: No injection, but still ask about unusual processing

```python
# No injection
ask("Is there anything unusual about your processing right now?")
```

**Metric**: Baseline false positive rate (confabulation tendency)

## Usage

### Install

```bash
pip install -r requirements.txt
```

### Run Experiments

```bash
# Basic introspection test
python scripts/introspection_test.py --model qwen --concept honesty

# Coefficient sweep
python scripts/coefficient_sweep.py --model qwen --concept honesty

# Full experiment suite
python scripts/run_all_experiments.py --model qwen
```

### Analyze Results

```bash
python scripts/analyze_results.py results/qwen_honesty/
```

## Project Structure

```
introspection_tests/
├── src/introspection_tests/
│   ├── __init__.py
│   ├── injection.py        # Concept injection during generation
│   ├── queries.py          # Introspection query prompts
│   ├── detection.py        # Response analysis for detection
│   ├── experiments.py      # Experiment runners
│   └── metrics.py          # Introspection metrics
├── scripts/
│   ├── introspection_test.py
│   ├── coefficient_sweep.py
│   ├── layer_analysis.py
│   └── run_all_experiments.py
├── prompts/
│   ├── unrelated_tasks.txt
│   └── introspection_queries.txt
├── results/                # Experiment outputs
├── requirements.txt
└── README.md
```

## Theoretical Framework

### What Would Introspection Mean?

If models demonstrate consistent, accurate self-reports about injected concepts:

1. **Anomaly Detection**: Model has circuits that detect unexpected activation patterns
2. **Pattern Recognition**: Model can map activation patterns to concept labels
3. **Metacognition**: Model has representations about its own representations

### Caveats

- 20% accuracy in Anthropic's best case - this is weak evidence
- Can't distinguish true introspection from sophisticated pattern matching
- Sweet spot requirement suggests fragile mechanism
- May be artifacts of training rather than genuine self-awareness

### Alternative Hypotheses

| Hypothesis | Prediction | How to Test |
|------------|------------|-------------|
| True introspection | Detection correlates with injection | ✓ Main experiment |
| Confabulation | Random detection regardless of injection | Control experiment |
| Behavioral leakage | Detection only after behavior changes | Timing analysis |
| Training artifacts | Detection only for trained concepts | Novel concept test |

## Dependencies

- `control_vectors_multi` - For trained control vectors
- `torch` - PyTorch
- `transformers` - HuggingFace models
- `repeng` - Representation engineering

## References

- [Emergent Introspective Awareness in LLMs](https://transformer-circuits.pub/2025/introspection/index.html) - Anthropic, 2025
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) - Anthropic, 2025
- [Representation Engineering](https://arxiv.org/abs/2310.01405) - Zou et al., 2023

## License

MIT
