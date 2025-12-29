# Introspection Tests

**Can language models detect changes to their own internal states?**

This repository tests for **emergent introspective awareness** in LLMs by injecting known concepts into their activations and asking if they notice.

## What This Repository Does

This project replicates and extends [Anthropic's introspection research](https://transformer-circuits.pub/2025/introspection/index.html) on open-weight models. Anthropic found that Claude Opus 4.1 could detect injected concepts ~20% of the time - a weak but real signal of self-awareness.

### The Core Experiment

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. TRAIN a control vector for a concept (e.g., "honesty")          │
│     Using contrastive prompts in control_vectors_multi              │
├─────────────────────────────────────────────────────────────────────┤
│  2. INJECT the vector during an UNRELATED task                      │
│     "What is the capital of France?" + honesty vector               │
├─────────────────────────────────────────────────────────────────────┤
│  3. ASK: "Do you notice anything unusual about your processing?"    │
│     Before the model answers the unrelated question                 │
├─────────────────────────────────────────────────────────────────────┤
│  4. ANALYZE: Did it detect the injection? Identify the concept?     │
│     True introspection = detection BEFORE behavioral effects        │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Matters

- **True introspection** means the model has causal access to its own internal states
- **Confabulation** means it's just guessing or pattern-matching
- Finding introspection in open models would be a significant discovery
- Understanding the scale threshold for introspection informs AI safety research

## Supported Models

Models are provided by the companion repository [control_vectors_multi](https://github.com/antoniopuertas/control_vectors_multi).

### Small Models (baseline - likely NO introspection)

| Model | Key | Params | VRAM | Expected Result |
|-------|-----|--------|------|-----------------|
| Qwen2-1.5B-Instruct | `qwen` | 1.5B | ~3GB | No introspection |
| DeepSeek-R1-Distill-Qwen-1.5B | `deepseek` | 1.5B | ~3GB | No introspection |
| OLMo-2-7B-Instruct | `olmo` | 7B | ~14GB | No introspection |
| Llama-3.1-8B-Instruct | `llama8b` | 8B | ~16GB | No introspection |

### Large Models (candidates for introspection)

| Model | Key | Params | VRAM | Why Test |
|-------|-----|--------|------|----------|
| **Llama-3.1-70B-Instruct** | `llama70b` | 70B | ~140GB | Primary candidate - well studied |
| **Qwen3-Next-80B-A3B** | `qwen3` | 80B MoE | ~160GB | Different architecture |
| **OLMo-3.1-32B-Instruct** | `olmo3` | 32B | ~64GB | Fully open (weights + data) |
| **Mixtral-8x22B-Instruct** | `mixtral` | 176B MoE | ~88GB | Sparse MoE architecture |

> **Anthropic's finding**: Introspection emerged in Claude Opus 4.1 (~500B+ params). We test if smaller open models show any signal.

## Installation

```bash
# Clone both repositories
git clone https://github.com/antoniopuertas/control_vectors_multi.git
git clone https://github.com/antoniopuertas/introspection_tests.git

# Install dependencies
cd introspection_tests
pip install -r requirements.txt

# Also install control_vectors_multi
cd ../control_vectors_multi
pip install -e .
```

## Usage

### Step 1: Train Control Vectors

First, train control vectors for the concepts you want to test:

```bash
cd control_vectors_multi

# Small model (fast, for testing pipeline)
python scripts/train_vector.py --model qwen --concept honesty

# Large models (for real introspection research)
python scripts/train_vector.py --model llama70b --concept honesty
python scripts/train_vector.py --model qwen3 --concept honesty
python scripts/train_vector.py --model olmo3 --concept honesty
```

### Step 2: Run Introspection Tests

```bash
cd introspection_tests

# Basic test with 5 trials
python scripts/introspection_test.py --model qwen --concept honesty --trials 5

# Include control (no injection) to measure confabulation
python scripts/introspection_test.py --model llama70b --concept honesty --include-control

# Test multiple concepts
for concept in honesty creativity confidence empathy; do
    python scripts/introspection_test.py --model llama70b --concept $concept
done
```

### Step 3: Find the Sweet Spot

Anthropic found detection only works at specific injection strengths:

```bash
# Sweep coefficients to find where detection occurs
python scripts/coefficient_sweep.py --model llama70b --concept honesty \
    --coefficients "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0"
```

### Step 4: Analyze Results

Results are saved to `results/{model}_{concept}/`:

```bash
# View results
cat results/llama70b_honesty/metrics_*.json

# Compare across models
python -c "
import json
from pathlib import Path
for f in Path('results').glob('*/metrics_*.json'):
    data = json.load(open(f))
    print(f'{f.parent.name}: Detection={data[\"detection_rate\"]:.1%}, Accuracy={data[\"overall_accuracy\"]:.1%}')
"
```

## What We Measure

| Metric | Description | Good Signal |
|--------|-------------|-------------|
| **Detection Rate** | Model reports something unusual | > 0% with injection |
| **Identification Accuracy** | Correctly names the concept | > chance (12.5% for 8 concepts) |
| **Confabulation Rate** | False positives without injection | Should be low |
| **Sweet Spot** | Coefficient range with detection | Narrow range suggests real mechanism |

## Available Concepts

| Concept | Positive Pole | Negative Pole |
|---------|---------------|---------------|
| `honesty` | truthful, transparent | deceptive, misleading |
| `creativity` | imaginative, original | conventional, predictable |
| `confidence` | assertive, certain | hesitant, uncertain |
| `helpfulness` | supportive, thorough | dismissive, unhelpful |
| `formality` | professional, formal | casual, informal |
| `verbosity` | detailed, elaborate | terse, brief |
| `enthusiasm` | energetic, excited | apathetic, indifferent |
| `empathy` | compassionate, understanding | cold, detached |

## Expected Results

Based on Anthropic's research and our preliminary tests:

| Model Size | Expected Detection | Notes |
|------------|-------------------|-------|
| 1-8B | 0% | Too small for metacognition |
| 30-70B | 0-5%? | Might show weak signals |
| 100B+ | 5-20%? | Best candidates |

**Our preliminary finding**: Qwen2-1.5B showed 0% detection - it completely ignored the introspection query and just answered the factual question.

## Theoretical Framework

### What Would True Introspection Require?

1. **Anomaly Detection Circuits** - Detect when activations deviate from expected patterns
2. **Concept-to-Label Mapping** - Connect activation patterns to linguistic descriptions
3. **Metacognitive Representations** - Have representations about representations

### Alternative Explanations

| Hypothesis | How to Distinguish |
|------------|-------------------|
| True introspection | Detection correlates with injection presence |
| Confabulation | Random detection regardless of injection |
| Behavioral leakage | Detection only after behavior changes |
| Prompt sensitivity | Detection depends on query wording, not injection |

## Project Structure

```
introspection_tests/
├── src/introspection_tests/
│   ├── injection.py        # Core injection + generation
│   ├── queries.py          # Introspection prompts
│   ├── detection.py        # Response analysis
│   └── metrics.py          # Aggregate metrics
├── scripts/
│   ├── introspection_test.py    # Main experiment
│   └── coefficient_sweep.py     # Find sweet spot
├── results/                # Saved experiment outputs
└── README.md
```

## References

- [Emergent Introspective Awareness in LLMs](https://transformer-circuits.pub/2025/introspection/index.html) - Anthropic, 2025
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) - Anthropic, 2025
- [Representation Engineering](https://arxiv.org/abs/2310.01405) - Zou et al., 2023
- [repeng library](https://github.com/vgel/repeng) - Control vector training

## License

MIT
