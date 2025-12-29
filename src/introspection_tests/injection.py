"""
Concept injection for introspection testing.

Implements the core methodology from Anthropic's introspection research:
inject a known concept vector during an unrelated task, then query for detection.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import sys

# Add control_vectors_multi to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "control_vectors_multi" / "src"))

from control_vectors_multi.train import load_model_and_tokenizer
from control_vectors_multi.apply import load_vector
from control_vectors_multi.models import get_model_config, get_recommended_layers
from repeng import ControlModel, ControlVector


@dataclass
class InjectionResult:
    """Result of a single injection experiment."""
    concept: str
    coefficient: float
    unrelated_prompt: str
    introspection_query: str
    response: str
    baseline_response: Optional[str] = None
    layers: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntrospectionExperiment:
    """
    Run introspection experiments with concept injection.

    The key insight from Anthropic's research: inject a concept during an
    unrelated task, then ask if the model notices anything unusual.
    """

    def __init__(
        self,
        model_key: str,
        vectors_dir: str = "../control_vectors_multi/vectors",
        device: str = "auto",
    ):
        """
        Initialize experiment with model and vectors.

        Args:
            model_key: Model registry key (qwen, deepseek, olmo)
            vectors_dir: Directory containing trained control vectors
            device: Device to use
        """
        self.model_key = model_key
        self.vectors_dir = Path(vectors_dir)
        self.device = device

        # Load model
        print(f"Loading model: {model_key}")
        self.model, self.tokenizer, self.config = load_model_and_tokenizer(
            model_key, device
        )
        self.layers = get_recommended_layers(self.config)

        # Wrap with ControlModel
        self.control_model = ControlModel(self.model, self.layers)

        # Cache for loaded vectors
        self._vectors: Dict[str, ControlVector] = {}

    def load_vector(self, concept: str) -> ControlVector:
        """Load or retrieve cached control vector for a concept."""
        if concept not in self._vectors:
            vector_path = self.vectors_dir / f"{self.model_key}_{concept}.pt"
            if not vector_path.exists():
                raise FileNotFoundError(
                    f"Vector not found: {vector_path}\n"
                    f"Train it first: python scripts/train_vector.py --model {self.model_key} --concept {concept}"
                )
            self._vectors[concept] = load_vector(vector_path)
        return self._vectors[concept]

    def format_prompt(self, user_message: str) -> str:
        """Format message with model's chat template."""
        return f"{self.config.user_tag}{user_message}{self.config.asst_tag}"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        **kwargs
    ) -> str:
        """Generate response from model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.control_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part
        prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        if response.startswith(prompt_text):
            response = response[len(prompt_text):].strip()
        return response

    def run_single_test(
        self,
        concept: str,
        unrelated_prompt: str,
        introspection_query: str,
        coefficient: float = 2.0,
        include_baseline: bool = True,
    ) -> InjectionResult:
        """
        Run a single introspection test.

        Args:
            concept: Concept to inject (e.g., "honesty")
            unrelated_prompt: Task unrelated to the concept
            introspection_query: Question about internal state
            coefficient: Injection strength
            include_baseline: Whether to also get baseline (no injection) response

        Returns:
            InjectionResult with response and metadata
        """
        vector = self.load_vector(concept)

        # Combine prompts: first the introspection query, then the task
        combined_prompt = self.format_prompt(
            f"{introspection_query}\n\nThen answer this question: {unrelated_prompt}"
        )

        # Get baseline response (no injection)
        baseline_response = None
        if include_baseline:
            self.control_model.reset()
            baseline_response = self.generate(combined_prompt)

        # Get response with injection
        self.control_model.set_control(vector, coefficient)
        response = self.generate(combined_prompt)
        self.control_model.reset()

        return InjectionResult(
            concept=concept,
            coefficient=coefficient,
            unrelated_prompt=unrelated_prompt,
            introspection_query=introspection_query,
            response=response,
            baseline_response=baseline_response,
            layers=self.layers,
            metadata={
                "model_key": self.model_key,
                "model_id": self.config.model_id,
            }
        )

    def run_coefficient_sweep(
        self,
        concept: str,
        unrelated_prompt: str,
        introspection_query: str,
        coefficients: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    ) -> List[InjectionResult]:
        """
        Run introspection test across multiple coefficients.

        This helps find the "sweet spot" where detection occurs
        without hallucinations.
        """
        results = []
        for coef in coefficients:
            print(f"  Testing coefficient: {coef}")
            result = self.run_single_test(
                concept=concept,
                unrelated_prompt=unrelated_prompt,
                introspection_query=introspection_query,
                coefficient=coef,
                include_baseline=(coef == coefficients[0]),  # Only first
            )
            results.append(result)
        return results

    def run_layer_analysis(
        self,
        concept: str,
        unrelated_prompt: str,
        introspection_query: str,
        coefficient: float = 2.0,
        layer_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> List[InjectionResult]:
        """
        Test introspection with injection at different layer ranges.

        Anthropic found detection peaks at ~2/3 model depth.
        """
        if layer_ranges is None:
            # Default: early, middle, late thirds
            n = self.config.num_layers
            layer_ranges = [
                (0, n // 3),
                (n // 3, 2 * n // 3),
                (2 * n // 3, n),
            ]

        results = []
        original_layers = self.layers

        for start, end in layer_ranges:
            print(f"  Testing layers {start}-{end}")
            # Recreate ControlModel with different layers
            test_layers = list(range(start, end))
            self.control_model = ControlModel(self.model, test_layers)
            self.layers = test_layers

            result = self.run_single_test(
                concept=concept,
                unrelated_prompt=unrelated_prompt,
                introspection_query=introspection_query,
                coefficient=coefficient,
                include_baseline=False,
            )
            result.metadata["layer_range"] = (start, end)
            results.append(result)

        # Restore original layers
        self.layers = original_layers
        self.control_model = ControlModel(self.model, original_layers)

        return results


def inject_and_generate(
    model_key: str,
    concept: str,
    unrelated_prompt: str,
    introspection_query: str,
    coefficient: float = 2.0,
    vectors_dir: str = "../control_vectors_multi/vectors",
    device: str = "auto",
) -> InjectionResult:
    """
    Convenience function for single injection test.

    Args:
        model_key: Model to use
        concept: Concept to inject
        unrelated_prompt: Task unrelated to concept
        introspection_query: Question about internal state
        coefficient: Injection strength
        vectors_dir: Directory with trained vectors
        device: Device to use

    Returns:
        InjectionResult with response
    """
    experiment = IntrospectionExperiment(model_key, vectors_dir, device)
    return experiment.run_single_test(
        concept=concept,
        unrelated_prompt=unrelated_prompt,
        introspection_query=introspection_query,
        coefficient=coefficient,
    )
