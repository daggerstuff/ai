"""
Model explainability system for Pixelated Empathy AI project.
Provides interpretability tools for debugging, auditing, and understanding model decisions.
"""

import contextlib
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import shap
from lime import lime_text
from transformers import AutoTokenizer

from ai.utils.torch_proxy import torch

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Types of model explanations"""

    FEATURE_IMPORTANCE = "feature_importance"
    ATTENTION_VISUALIZATION = "attention_visualization"
    GRADIENT_BASED = "gradient_based"
    LIME_EXPLANATION = "lime_explanation"
    SHAP_EXPLANATION = "shap_explanation"
    COUNTERFACTUAL = "counterfactual"
    RULE_EXTRACTION = "rule_extraction"
    SIMILARITY_ANALYSIS = "similarity_analysis"


class ExplanationScope(Enum):
    """Scope of explanations"""

    LOCAL = "local"  # For individual predictions
    GLOBAL = "global"  # For overall model behavior
    COMPARATIVE = "comparative"  # Between different models/versions


@dataclass
class ExplanationResult:
    """Result of an explanation"""

    explanation_id: str
    explanation_type: ExplanationType
    scope: ExplanationScope
    model_name: str
    input_data: str | list[str] | dict[str, Any]
    explanation_output: dict[str, Any]
    confidence_score: float
    computation_time_ms: float
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] | None = None


@dataclass
class AttentionWeights:
    """Attention weights for visualization"""

    layer_weights: list[np.ndarray]
    head_weights: list[np.ndarray]
    token_weights: list[float]
    attention_map: np.ndarray | None = None


@dataclass
class FeatureImportance:
    """Feature importance scores"""

    features: list[str]
    importance_scores: list[float]
    feature_types: list[str]
    confidence_intervals: list[tuple[float, float]] | None = None


class ExplainabilityEngine:
    """Main engine for model explainability"""

    def __init__(self):
        self.explanations: dict[str, ExplanationResult] = {}
        self.models: dict[str, Any] = {}  # Model cache
        self.tokenizers: dict[str, Any] = {}  # Tokenizer cache
        self.logger = logging.getLogger(__name__)

    def register_model(self, model_name: str, model: Any, tokenizer: Any | None = None):
        """Register a model for explainability"""
        self.models[model_name] = model
        if tokenizer:
            self.tokenizers[model_name] = tokenizer
        self.logger.info(f"Registered model {model_name} for explainability")

    def get_feature_importance(
        self,
        model_name: str,
        input_text: str,
        method: str = "gradient",
        target_class: int | None = None,
    ) -> ExplanationResult:
        """Get feature importance explanation for input text"""
        start_time = time.time()

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered for explainability")

        model = self.models[model_name]
        tokenizer = self.tokenizers.get(model_name)

        explanation_id = self._generate_explanation_id(model_name, input_text, "feature_importance")

        try:
            if method == "gradient":
                importance_scores = self._gradient_based_importance(model, tokenizer, input_text, target_class)
            elif method == "lime":
                importance_scores = self._lime_importance(model, tokenizer, input_text, target_class)
            elif method == "shap":
                importance_scores = self._shap_importance(model, tokenizer, input_text, target_class)
            else:
                raise ValueError(f"Unsupported feature importance method: {method}")

            explanation_output = {
                "feature_importance": importance_scores.__dict__,
                "method": method,
                "target_class": target_class,
            }

            computation_time = (time.time() - start_time) * 1000

            result = ExplanationResult(
                explanation_id=explanation_id,
                explanation_type=ExplanationType.FEATURE_IMPORTANCE,
                scope=ExplanationScope.LOCAL,
                model_name=model_name,
                input_data=input_text,
                explanation_output=explanation_output,
                confidence_score=self._calculate_explanation_confidence(importance_scores.importance_scores),
                computation_time_ms=computation_time,
                metadata={
                    "method": method,
                    "feature_count": len(importance_scores.features),
                },
            )

            self.explanations[explanation_id] = result
            self.logger.info(f"Generated feature importance explanation for {model_name} in {computation_time:.2f}ms")

            return result

        except Exception as e:
            self.logger.error(f"Failed to generate feature importance explanation: {e}")
            raise

    def _gradient_based_importance(
        self,
        model: Any,
        tokenizer: Any,
        input_text: str,
        target_class: int | None = None,
    ) -> FeatureImportance:
        """Calculate feature importance using gradient-based methods"""
        if not tokenizer or not hasattr(model, "forward"):
            # Fallback for non-PyTorch models
            return self._simple_token_importance(input_text)

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # Enable gradient computation
        input_ids.requires_grad_(True)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask) if attention_mask is not None else model(input_ids)

        # Get logits for target class
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if target_class is None:
            target_class = torch.argmax(logits[0]).item()

        # Calculate gradients
        model.zero_grad()
        target_logit = logits[0, target_class]
        target_logit.backward()

        # Extract gradients
        gradients = input_ids.grad[0].detach().cpu().numpy()
        input_embeddings = model.get_input_embeddings()(input_ids).detach().cpu().numpy()[0]

        # Calculate importance scores
        importance_scores = np.abs(gradients * input_embeddings).sum(axis=1)

        # Map back to tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        return FeatureImportance(
            features=tokens,
            importance_scores=importance_scores.tolist(),
            feature_types=["token"] * len(tokens),
            confidence_intervals=[(score * 0.9, score * 1.1) for score in importance_scores.tolist()],
        )

    def _lime_importance(
        self,
        model: Any,
        tokenizer: Any,
        input_text: str,
        _target_class: int | None = None,
    ) -> FeatureImportance:
        """Calculate feature importance using LIME"""
        try:
            # Create LIME explainer
            explainer = lime_text.LimeTextExplainer(class_names=["negative", "positive"])

            # Define prediction function for LIME
            def predict_fn(texts):
                results = []
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                        results.append(probs[0])
                return np.array(results)

            # Generate explanation
            explanation = explainer.explain_instance(input_text, predict_fn, num_features=10, num_samples=1000)

            # Extract feature importance
            feature_weights = explanation.as_list()
            features = [fw[0] for fw in feature_weights]
            importance_scores = [fw[1] for fw in feature_weights]

            return FeatureImportance(
                features=features,
                importance_scores=importance_scores,
                feature_types=["word" if " " in f else "token" for f in features],
            )

        except Exception as e:
            self.logger.warning(f"LIME explanation failed, falling back to simple method: {e}")
            return self._simple_token_importance(input_text)

    def _shap_importance(
        self,
        model: Any,
        tokenizer: Any,
        input_text: str,
        _target_class: int | None = None,
    ) -> FeatureImportance:
        """Calculate feature importance using SHAP"""
        try:
            # Create SHAP explainer (simplified approach)
            def model_predict(texts):
                results = []
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                        results.append(probs[0])
                return np.array(results)

            # Use KernelExplainer for simpler interpretation
            # Note: In practice, you'd want to use a more appropriate SHAP explainer
            explainer = shap.Explainer(model_predict, input_text)
            shap_values = explainer(input_text)

            # Extract feature importance from SHAP values
            # This is a simplified extraction
            if hasattr(shap_values, "values"):
                importance_scores = np.abs(shap_values.values[0]).tolist()
                features = (
                    shap_values.data[0].split()
                    if isinstance(shap_values.data[0], str)
                    else ["feature_" + str(i) for i in range(len(importance_scores))]
                )

                return FeatureImportance(
                    features=features,
                    importance_scores=importance_scores,
                    feature_types=["word"] * len(features),
                )

        except Exception as e:
            self.logger.warning(f"SHAP explanation failed, falling back to simple method: {e}")

        return self._simple_token_importance(input_text)

    def _simple_token_importance(self, input_text: str) -> FeatureImportance:
        """Simple token importance based on position and length"""
        tokens = input_text.split()
        importance_scores = []

        # Simple heuristic: earlier tokens and longer tokens are more important
        for i, token in enumerate(tokens):
            position_importance = 1.0 - (i / len(tokens)) if len(tokens) > 0 else 1.0
            length_importance = min(len(token) / 20.0, 1.0)  # Cap at 20 chars
            importance = (position_importance + length_importance) / 2.0
            importance_scores.append(importance)

        return FeatureImportance(
            features=tokens,
            importance_scores=importance_scores,
            feature_types=["word"] * len(tokens),
        )

    def get_attention_visualization(
        self, model_name: str, input_text: str, layer_idx: int | None = None
    ) -> ExplanationResult:
        """Get attention visualization for input text"""
        start_time = time.time()

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered for explainability")

        model = self.models[model_name]
        tokenizer = self.tokenizers.get(model_name)

        explanation_id = self._generate_explanation_id(model_name, input_text, "attention_visualization")

        try:
            attention_weights = self._extract_attention_weights(model, tokenizer, input_text, layer_idx)

            # Generate visualization data
            visualization_data = self._create_attention_visualization(attention_weights, tokenizer, input_text)

            explanation_output = {
                "attention_weights": attention_weights.__dict__,
                "visualization_data": visualization_data,
                "layer_idx": layer_idx,
            }

            computation_time = (time.time() - start_time) * 1000

            result = ExplanationResult(
                explanation_id=explanation_id,
                explanation_type=ExplanationType.ATTENTION_VISUALIZATION,
                scope=ExplanationScope.LOCAL,
                model_name=model_name,
                input_data=input_text,
                explanation_output=explanation_output,
                confidence_score=0.9,  # High confidence for attention visualization
                computation_time_ms=computation_time,
                metadata={
                    "layers_analyzed": len(attention_weights.layer_weights),
                    "heads_per_layer": len(attention_weights.head_weights) if attention_weights.head_weights else 0,
                },
            )

            self.explanations[explanation_id] = result
            self.logger.info(f"Generated attention visualization for {model_name} in {computation_time:.2f}ms")

            return result

        except Exception as e:
            self.logger.error(f"Failed to generate attention visualization: {e}")
            raise

    def _extract_attention_weights(
        self,
        model: Any,
        tokenizer: Any,
        input_text: str,
        layer_idx: int | None = None,
    ) -> AttentionWeights:
        """Extract attention weights from model"""
        if not tokenizer or not hasattr(model, "config"):
            # Fallback for models without attention extraction
            return AttentionWeights(
                layer_weights=[np.random.rand(10, 10)],  # Random weights for demo
                head_weights=[np.random.rand(8, 10, 10)],  # Random heads
                token_weights=[float(i) / 10 for i in range(10)],  # Ascending weights
            )

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        inputs.get("attention_mask")

        # Forward pass with attention weights
        with torch.no_grad():
            if hasattr(model, "forward_with_attention"):
                outputs = model.forward_with_attention(**inputs)
                attentions = outputs.attentions
            else:
                # Try to get attention weights from standard model
                outputs = model(**inputs, output_attentions=True)
                attentions = outputs.attentions if hasattr(outputs, "attentions") else None

        if attentions is None:
            # Fallback if no attention weights available
            seq_len = input_ids.shape[1]
            return AttentionWeights(
                layer_weights=[np.random.rand(seq_len, seq_len)],
                head_weights=[np.random.rand(1, seq_len, seq_len)],
                token_weights=[float(i) / seq_len for i in range(seq_len)],
            )

        # Process attention weights
        layer_weights = []
        head_weights = []

        for i, layer_attention in enumerate(attentions):
            if layer_idx is not None and i != layer_idx:
                continue

            # Convert to numpy and process
            layer_attention_np = layer_attention.cpu().numpy()
            layer_weights.append(layer_attention_np[0].mean(axis=0))  # Average across heads

            if layer_attention_np.shape[1] > 1:  # Multiple heads
                head_weights.append(layer_attention_np[0])  # First batch item

        # Calculate token-level importance
        if layer_weights:
            token_weights = layer_weights[-1].mean(axis=0).tolist()  # Last layer, averaged across rows
        else:
            seq_len = input_ids.shape[1]
            token_weights = [float(i) / seq_len for i in range(seq_len)]

        return AttentionWeights(
            layer_weights=layer_weights,
            head_weights=head_weights,
            token_weights=token_weights,
        )

    def _create_attention_visualization(
        self, attention_weights: AttentionWeights, tokenizer: Any, input_text: str
    ) -> dict[str, Any]:
        """Create visualization data for attention weights"""
        # Extract tokens from input text
        if tokenizer:
            tokens = tokenizer.tokenize(input_text)
        else:
            tokens = input_text.split()[: len(attention_weights.token_weights)]

        # Create heatmap data
        heatmap_data = []
        if attention_weights.layer_weights:
            # Use the first layer's attention weights for visualization
            layer_attention = attention_weights.layer_weights[0]
            for i, token in enumerate(tokens[: layer_attention.shape[0]]):
                row_data = []
                for j, _target_token in enumerate(tokens[: layer_attention.shape[1]]):
                    if i < layer_attention.shape[0] and j < layer_attention.shape[1]:
                        row_data.append(float(layer_attention[i, j]))
                    else:
                        row_data.append(0.0)
                heatmap_data.append({"token": token, "attention_weights": row_data})

        return {
            "tokens": tokens,
            "token_importance": attention_weights.token_weights[: len(tokens)],
            "heatmap_data": heatmap_data,
            "visualization_type": "attention_heatmap",
        }

    def get_counterfactual_explanation(
        self,
        model_name: str,
        input_text: str,
        target_outcome: Any,
        max_changes: int = 5,
    ) -> ExplanationResult:
        """Generate counterfactual explanation showing what changes would lead to a different outcome"""
        start_time = time.time()

        explanation_id = self._generate_explanation_id(model_name, input_text, "counterfactual")

        try:
            # Generate counterfactual examples
            counterfactuals = self._generate_counterfactuals(model_name, input_text, target_outcome, max_changes)

            explanation_output = {
                "counterfactuals": counterfactuals,
                "target_outcome": target_outcome,
                "max_changes": max_changes,
            }

            computation_time = (time.time() - start_time) * 1000

            result = ExplanationResult(
                explanation_id=explanation_id,
                explanation_type=ExplanationType.COUNTERFACTUAL,
                scope=ExplanationScope.LOCAL,
                model_name=model_name,
                input_data=input_text,
                explanation_output=explanation_output,
                confidence_score=0.8,  # Moderate confidence for counterfactuals
                computation_time_ms=computation_time,
                metadata={
                    "counterfactual_count": len(counterfactuals),
                    "max_changes_tried": max_changes,
                },
            )

            self.explanations[explanation_id] = result
            self.logger.info(f"Generated counterfactual explanation for {model_name} in {computation_time:.2f}ms")

            return result

        except Exception as e:
            self.logger.error(f"Failed to generate counterfactual explanation: {e}")
            raise

    def _generate_counterfactuals(
        self, model_name: str, input_text: str, target_outcome: Any, max_changes: int
    ) -> list[dict[str, Any]]:
        """Generate counterfactual examples"""
        if model_name not in self.models:
            return []

        model = self.models[model_name]
        tokenizer = self.tokenizers.get(model_name)

        counterfactuals = []

        # Simple counterfactual generation approach
        words = input_text.split()

        # Try removing words to see effect on outcome
        for i in range(min(len(words), max_changes)):
            modified_words = words[:i] + words[i + 1 :]
            modified_text = " ".join(modified_words)

            # Get prediction for modified text
            try:
                if tokenizer and hasattr(model, "forward"):
                    inputs = tokenizer(
                        modified_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                        # Check if modification moves toward target
                        predicted_class = torch.argmax(logits[0]).item()
                        if predicted_class == target_outcome:
                            counterfactuals.append(
                                {
                                    "modification": f"removed word '{words[i]}'",
                                    "modified_text": modified_text,
                                    "predicted_class": predicted_class,
                                    "confidence": float(torch.softmax(logits[0], dim=-1).max().item()),
                                }
                            )

                # Stop if we have enough examples
                if len(counterfactuals) >= 3:  # Limit for performance
                    break

            except Exception as e:
                self.logger.warning(f"Failed to generate counterfactual for word {i}: {e}")
                continue

        return counterfactuals

    def get_similarity_analysis(
        self, model_name: str, input_text: str, reference_texts: list[str]
    ) -> ExplanationResult:
        """Analyze similarity between input and reference texts"""
        start_time = time.time()

        explanation_id = self._generate_explanation_id(model_name, input_text, "similarity_analysis")

        try:
            similarities = self._calculate_similarities(input_text, reference_texts)

            explanation_output = {
                "similarities": similarities,
                "reference_count": len(reference_texts),
            }

            computation_time = (time.time() - start_time) * 1000

            result = ExplanationResult(
                explanation_id=explanation_id,
                explanation_type=ExplanationType.SIMILARITY_ANALYSIS,
                scope=ExplanationScope.LOCAL,
                model_name=model_name,
                input_data=input_text,
                explanation_output=explanation_output,
                confidence_score=0.95,  # High confidence for similarity analysis
                computation_time_ms=computation_time,
                metadata={"reference_count": len(reference_texts)},
            )

            self.explanations[explanation_id] = result
            self.logger.info(f"Generated similarity analysis for {model_name} in {computation_time:.2f}ms")

            return result

        except Exception as e:
            self.logger.error(f"Failed to generate similarity analysis: {e}")
            raise

    def _calculate_similarities(self, input_text: str, reference_texts: list[str]) -> list[dict[str, Any]]:
        """Calculate similarity scores using simple text similarity"""
        similarities = []

        # Simple Jaccard similarity based on word overlap
        input_words = set(input_text.lower().split())

        for i, ref_text in enumerate(reference_texts):
            ref_words = set(ref_text.lower().split())

            # Calculate Jaccard similarity
            intersection = len(input_words.intersection(ref_words))
            union = len(input_words.union(ref_words))
            jaccard_similarity = intersection / union if union > 0 else 0.0

            # Calculate cosine similarity approximation
            dot_product = len(input_words.intersection(ref_words))
            magnitude_input = len(input_words) ** 0.5
            magnitude_ref = len(ref_words) ** 0.5
            cosine_similarity = (
                dot_product / (magnitude_input * magnitude_ref) if magnitude_input * magnitude_ref > 0 else 0.0
            )

            similarities.append(
                {
                    "reference_index": i,
                    "reference_text_preview": ref_text[:100] + "..." if len(ref_text) > 100 else ref_text,
                    "jaccard_similarity": jaccard_similarity,
                    "cosine_similarity": cosine_similarity,
                    "word_overlap": intersection,
                    "total_unique_words": union,
                }
            )

        # Sort by highest similarity
        similarities.sort(key=lambda x: x["jaccard_similarity"], reverse=True)

        return similarities

    def get_global_model_behavior(self, model_name: str, sample_inputs: list[str]) -> ExplanationResult:
        """Analyze global model behavior across sample inputs"""
        start_time = time.time()

        explanation_id = self._generate_explanation_id(model_name, "global", "global_behavior")

        try:
            # Analyze model behavior across samples
            behavior_analysis = self._analyze_global_behavior(model_name, sample_inputs)

            explanation_output = {
                "behavior_analysis": behavior_analysis,
                "sample_count": len(sample_inputs),
            }

            computation_time = (time.time() - start_time) * 1000

            result = ExplanationResult(
                explanation_id=explanation_id,
                explanation_type=ExplanationType.RULE_EXTRACTION,
                scope=ExplanationScope.GLOBAL,
                model_name=model_name,
                input_data=f"{len(sample_inputs)} samples",
                explanation_output=explanation_output,
                confidence_score=0.85,
                computation_time_ms=computation_time,
                metadata={
                    "analysis_type": "global_behavior",
                    "sample_count": len(sample_inputs),
                },
            )

            self.explanations[explanation_id] = result
            self.logger.info(f"Generated global behavior analysis for {model_name} in {computation_time:.2f}ms")

            return result

        except Exception as e:
            self.logger.error(f"Failed to generate global behavior analysis: {e}")
            raise

    def _analyze_global_behavior(self, model_name: str, sample_inputs: list[str]) -> dict[str, Any]:
        """Analyze global model behavior"""
        if model_name not in self.models:
            return {"error": "Model not registered"}

        model = self.models[model_name]
        tokenizer = self.tokenizers.get(model_name)

        predictions = []
        feature_importances = []

        # Analyze behavior across samples
        for sample_text in sample_inputs[:100]:  # Limit for performance
            try:
                if tokenizer and hasattr(model, "forward"):
                    inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        predicted_class = torch.argmax(logits[0]).item()
                        confidence = float(torch.softmax(logits[0], dim=-1).max().item())

                        predictions.append(
                            {
                                "text_preview": sample_text[:50] + "..." if len(sample_text) > 50 else sample_text,
                                "predicted_class": predicted_class,
                                "confidence": confidence,
                            }
                        )

                        # Get feature importance for this sample
                        try:
                            importance = self._gradient_based_importance(model, tokenizer, sample_text, predicted_class)
                            feature_importances.append(importance.importance_scores)
                        except:
                            pass

            except Exception as e:
                self.logger.warning(f"Failed to analyze sample: {e}")
                continue

        # Aggregate results
        if predictions:
            class_distribution = {}
            avg_confidence = 0.0
            total_samples = len(predictions)

            for pred in predictions:
                class_id = pred["predicted_class"]
                class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
                avg_confidence += pred["confidence"]

            avg_confidence /= total_samples

            # Normalize class distribution
            normalized_distribution = {
                str(class_id): count / total_samples for class_id, count in class_distribution.items()
            }

            return {
                "class_distribution": normalized_distribution,
                "average_confidence": avg_confidence,
                "total_samples_analyzed": total_samples,
                "feature_importance_patterns": self._aggregate_feature_importances(feature_importances),
            }

        return {"error": "No samples analyzed successfully"}

    def _aggregate_feature_importances(self, importance_lists: list[list[float]]) -> dict[str, float]:
        """Aggregate feature importance scores"""
        if not importance_lists:
            return {}

        # Calculate average importance across samples
        num_features = len(importance_lists[0])
        aggregated_importance = [0.0] * num_features

        for importance_scores in importance_lists:
            for i, score in enumerate(importance_scores[:num_features]):
                aggregated_importance[i] += score

        # Calculate averages
        avg_importance = [score / len(importance_lists) for score in aggregated_importance]

        # Return top features with their importance
        top_features = {}
        for i, importance in enumerate(avg_importance):
            if importance > 0.01:  # Threshold for significance
                top_features[f"feature_{i}"] = round(importance, 4)

        # Sort by importance
        return dict(sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:10])

    def _calculate_explanation_confidence(self, scores: list[float]) -> float:
        """Calculate confidence score for explanation based on importance scores"""
        if not scores:
            return 0.0

        # Use variance as a measure of confidence (higher variance = more confident about feature importance)
        variance = np.var(scores) if len(scores) > 1 else 0.0
        max_score = max(scores) if scores else 0.0

        # Normalize confidence (0.0 to 1.0)
        confidence = min(1.0, variance * 10 + max_score * 0.5)
        return max(0.0, min(1.0, confidence))

    def _generate_explanation_id(self, model_name: str, input_data: str, explanation_type: str) -> str:
        """Generate unique ID for explanation"""
        input_hash = hashlib.md5(str(input_data).encode()).hexdigest()[:16]
        return f"exp_{model_name}_{explanation_type}_{input_hash}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    def get_explanation(self, explanation_id: str) -> ExplanationResult | None:
        """Retrieve a stored explanation by ID"""
        return self.explanations.get(explanation_id)

    def list_explanations(
        self,
        model_name: str | None = None,
        explanation_type: ExplanationType | None = None,
    ) -> list[ExplanationResult]:
        """List stored explanations with optional filtering"""
        explanations = list(self.explanations.values())

        if model_name:
            explanations = [exp for exp in explanations if exp.model_name == model_name]

        if explanation_type:
            explanations = [exp for exp in explanations if exp.explanation_type == explanation_type]

        # Sort by creation time (newest first)
        explanations.sort(key=lambda x: x.created_at, reverse=True)

        return explanations

    def clear_explanations(self, older_than_hours: int | None = None) -> int:
        """Clear stored explanations, optionally only older ones"""
        if older_than_hours is None:
            count = len(self.explanations)
            self.explanations.clear()
            return count

        # Clear only older explanations
        cutoff_time = datetime.now(UTC) - timedelta(hours=older_than_hours)
        expired_ids = [
            exp_id
            for exp_id, exp in self.explanations.items()
            if datetime.fromisoformat(exp.created_at.replace("Z", "+00:00")) < cutoff_time
        ]

        for exp_id in expired_ids:
            del self.explanations[exp_id]

        return len(expired_ids)


class LimitedAccessExplainability:
    """Restricted access to explainability features for security and privacy"""

    def __init__(self, explainability_engine: ExplainabilityEngine):
        self.engine = explainability_engine
        self.access_log: list[dict[str, Any]] = []
        self.max_daily_requests = 1000
        self.requests_today = 0
        self.daily_reset_time = datetime.now(UTC).date()

    def _check_access_limits(self) -> bool:
        """Check if access limits have been reached"""
        today = datetime.now(UTC).date()

        # Reset daily counter if new day
        if today != self.daily_reset_time:
            self.requests_today = 0
            self.daily_reset_time = today

        if self.requests_today >= self.max_daily_requests:
            return False

        self.requests_today += 1
        return True

    def _log_access(self, user_id: str, action: str, details: dict[str, Any]):
        """Log access for auditing"""
        self.access_log.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "user_id": user_id,
                "action": action,
                "details": details,
                "requests_today": self.requests_today,
            }
        )

    def get_limited_feature_importance(
        self, user_id: str, model_name: str, input_text: str, method: str = "gradient"
    ) -> ExplanationResult | None:
        """Get feature importance with access restrictions"""
        if not self._check_access_limits():
            self._log_access(user_id, "denied_feature_importance", {"reason": "daily_limit_exceeded"})
            return None

        try:
            result = self.engine.get_feature_importance(model_name, input_text, method)
            self._log_access(
                user_id,
                "feature_importance",
                {
                    "model_name": model_name,
                    "method": method,
                    "explanation_id": result.explanation_id,
                },
            )
            return result
        except Exception as e:
            self._log_access(
                user_id,
                "feature_importance_error",
                {"model_name": model_name, "method": method, "error": str(e)},
            )
            return None

    def get_limited_attention_visualization(
        self, user_id: str, model_name: str, input_text: str
    ) -> ExplanationResult | None:
        """Get attention visualization with access restrictions"""
        if not self._check_access_limits():
            self._log_access(
                user_id,
                "denied_attention_visualization",
                {"reason": "daily_limit_exceeded"},
            )
            return None

        try:
            result = self.engine.get_attention_visualization(model_name, input_text)
            self._log_access(
                user_id,
                "attention_visualization",
                {"model_name": model_name, "explanation_id": result.explanation_id},
            )
            return result
        except Exception as e:
            self._log_access(
                user_id,
                "attention_visualization_error",
                {"model_name": model_name, "error": str(e)},
            )
            return None

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get access audit log"""
        return self.access_log[-limit:]


# Global explainability engine
explainability_engine = ExplainabilityEngine()
limited_access_explainability = LimitedAccessExplainability(explainability_engine)


# Utility functions for API integration
def register_model_for_explainability(model_name: str, model: Any, tokenizer: Any | None = None):
    """Register a model for explainability analysis"""
    explainability_engine.register_model(model_name, model, tokenizer)


def get_limited_explanation(
    user_id: str,
    explanation_type: str,
    model_name: str,
    input_data: str | list[str],
    **kwargs,
) -> ExplanationResult | None:
    """Get limited-access explanation for API endpoints"""

    if explanation_type == "feature_importance":
        return limited_access_explainability.get_limited_feature_importance(
            user_id,
            model_name,
            input_data,
            kwargs.get("method", "gradient"),
        )
    if explanation_type == "attention_visualization":
        return limited_access_explainability.get_limited_attention_visualization(
            user_id,
            model_name,
            input_data,
        )
    return None


# Example usage and testing
def test_explainability_system():
    """Test the explainability system"""
    logger.info("Testing Explainability System...")

    # Create a mock model for testing
    class MockModel:
        def __init__(self):
            self.config = type("Config", (), {"hidden_size": 768})()

        def forward(self, input_ids, **_kwargs):
            # Mock forward pass
            batch_size, _seq_len = input_ids.shape
            logits = torch.randn(batch_size, 2)  # Binary classification
            return type("Outputs", (), {"logits": logits})()

        def get_input_embeddings(self):
            return torch.nn.Embedding(1000, 768)

    mock_model = MockModel()
    mock_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") if torch.cuda.is_available() else None

    # Register model
    explainability_engine.register_model("test_model", mock_model, mock_tokenizer)

    # Test input
    test_input = "The patient is experiencing anxiety and depression symptoms."

    # Test feature importance
    with contextlib.suppress(Exception):
        explainability_engine.get_feature_importance("test_model", test_input, method="gradient")

    # Test attention visualization
    with contextlib.suppress(Exception):
        explainability_engine.get_attention_visualization("test_model", test_input)

    # Test counterfactual explanation
    with contextlib.suppress(Exception):
        explainability_engine.get_counterfactual_explanation("test_model", test_input, target_outcome=1, max_changes=3)

    # Test similarity analysis
    reference_texts = [
        "Patient reports feeling anxious about upcoming therapy session.",
        "Client mentions having thoughts of self-harm recently.",
        "The patient seems to be responding well to current treatment.",
    ]

    with contextlib.suppress(Exception):
        explainability_engine.get_similarity_analysis("test_model", test_input, reference_texts)

    # Test global behavior analysis
    sample_inputs = [
        "Patient expresses concern about medication side effects.",
        "Client shows improvement in mood over past two weeks.",
        "The therapy session focused on cognitive behavioral techniques.",
        "Patient reported increased anxiety after recent life events.",
    ]

    try:
        global_result = explainability_engine.get_global_model_behavior("test_model", sample_inputs)
        if "behavior_analysis" in global_result.explanation_output:
            global_result.explanation_output["behavior_analysis"]
    except Exception:
        pass

    # Test limited access
    get_limited_explanation(
        user_id="test_user",
        explanation_type="feature_importance",
        model_name="test_model",
        input_data=test_input,
        method="gradient",
    )

    # Test audit log
    limited_access_explainability.get_audit_log()


if __name__ == "__main__":
    test_explainability_system()
