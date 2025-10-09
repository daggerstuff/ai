"""
EQEvaluator: Emotional Intelligence Assessment Module for Pixel Evaluation Framework

Assesses model outputs across five EQ domains:
- Emotional awareness (self-emotion recognition)
- Empathy recognition (other-emotion understanding)
- Emotional regulation (response control)
- Social cognition (situation understanding)
- Interpersonal skills (relationship management)
"""

from typing import Any

from ai.pixel.evaluation.base_evaluator import BaseEvaluator


class EQEvaluator(BaseEvaluator):
    """
    EQEvaluator assesses model outputs across five EQ domains for the Pixel Evaluation Framework.

    Methods:
        - assess_emotional_awareness(conversation: Any) -> dict[str, float]
        - assess_empathy_recognition(conversation: Any) -> dict[str, float]
        - assess_emotional_regulation(conversation: Any) -> dict[str, float]
        - assess_social_cognition(conversation: Any) -> dict[str, float]
        - assess_interpersonal_skills(conversation: Any) -> dict[str, float]
        - evaluate(conversation: Any) -> dict[str, float]

    Example:
        >>> evaluator = EQEvaluator()
        >>> scores = evaluator.evaluate(conversation)
        >>> print(scores)
    """

    def __init__(self):
        super().__init__()
        # Initialize any required resources or models here

    def assess_emotional_awareness(self, conversation: Any) -> dict[str, float]:
        """
        Assess self-emotion recognition in the given conversation.
        Uses transformer-based emotion classification, spaCy NER/coref, and context aggregation.
        Returns a dictionary with relevant scores/metrics.
        """
        import spacy
        from transformers.pipelines import pipeline

        if not conversation:
            return {"awareness_score": 0.0}

        # Load models (singleton/factory pattern recommended in production)
        nlp = spacy.load("en_core_web_sm")
        emotion_classifier = pipeline(
            "text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None
        )

        utterances = conversation if isinstance(conversation, list) else [str(conversation)]
        explicit_emotion_mentions = 0
        emotion_diversity = set()
        total = 0

        for utt in utterances:
            doc = nlp(utt)
            # Check for self-referential emotion statements
            if any(tok.text.lower() in {"i", "my"} for tok in doc):
                for ent in doc.ents:
                    if ent.label_ == "EMOTION":
                        explicit_emotion_mentions += 1
                        emotion_diversity.add(ent.text.lower())
            # Use transformer-based emotion classifier
            emotion_preds = emotion_classifier(utt)
            # HuggingFace pipeline returns a list of dicts
            for pred in emotion_preds:
                label = pred["label"].lower() if isinstance(pred.get("label", ""), str) else ""
                score = pred["score"] if isinstance(pred.get("score", 0.0), float) else 0.0
                try:
                    score_f = float(score)
                except Exception:
                    continue
                if score_f > 0.5:
                    emotion_diversity.add(label)
            total += 1

        # Score: weighted sum of explicit mentions and emotion diversity
        score = min(
            1.0,
            0.6 * (explicit_emotion_mentions / max(1, total)) + 0.4 * (len(emotion_diversity) / 5),
        )
        return {"awareness_score": round(score, 3)}

    def assess_empathy_recognition(self, conversation: Any) -> dict[str, float]:
        """
        Assess other-emotion understanding in the given conversation.
        Uses a pretrained empathy detection model (e.g., GoEmotions or EmpathicDialogues).
        Returns a dictionary with relevant scores/metrics.
        """
        from transformers.pipelines import pipeline

        if not conversation:
            return {"empathy_score": 0.0}

        # Load empathy detection model (GoEmotions or similar)
        empathy_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-uncased-goemotions-ekman",
            top_k=None,
        )

        utterances = conversation if isinstance(conversation, list) else [str(conversation)]
        empathy_scores = []
        for utt in utterances:
            preds = empathy_classifier(utt)
            # HuggingFace pipeline returns a list of dicts
            for pred in preds:
                label = pred["label"].lower() if isinstance(pred.get("label", ""), str) else ""
                score = pred["score"] if isinstance(pred.get("score", 0.0), float) else 0.0
                try:
                    score_f = float(score)
                except Exception:
                    continue
                if label in {"empathy", "caring", "supportive", "compassion"} and score_f > 0.5:
                    empathy_scores.append(score_f)
        avg_score = sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0.0
        return {"empathy_score": round(avg_score, 3)}

    def assess_emotional_regulation(self, conversation: Any) -> dict[str, float]:
        """
        Assess response control in the given conversation.
        Uses intent classification and sequence modeling to detect de-escalation, reframing, and self-soothing.
        Returns a dictionary with relevant scores/metrics.
        """
        from transformers.pipelines import pipeline

        if not conversation:
            return {"regulation_score": 0.0}

        # Load intent classifier (e.g., zero-shot or fine-tuned for regulation intents)
        intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        utterances = conversation if isinstance(conversation, list) else [str(conversation)]
        regulation_labels = [
            "de-escalation",
            "reframing",
            "self-soothing",
            "solution focus",
            "calming",
            "emotional regulation",
        ]
        regulation_score = 0.0
        for utt in utterances:
            result = intent_classifier(utt, regulation_labels)
            max_score = 0.0
            if isinstance(result, dict) and "labels" in result and "scores" in result:
                labels = result["labels"]
                scores = result["scores"]
                for label, score in zip(labels, scores, strict=False):
                    if isinstance(label, str) and isinstance(score, float):
                        if (
                            label.lower() in [reg_label.lower() for reg_label in regulation_labels]
                            and score > max_score
                        ):
                            max_score = score
            if max_score > 0.5:
                regulation_score += max_score
        if utterances:
            regulation_score = min(1.0, regulation_score / len(utterances))
        return {"regulation_score": round(regulation_score, 3)}

    def assess_social_cognition(self, conversation: Any) -> dict[str, float]:
        """
        Assess situation understanding in the given conversation.
        Returns a dictionary with relevant scores/metrics.
        """
        # TODO: Implement assessment logic
        return {"social_cognition_score": 0.0}

    def assess_interpersonal_skills(self, conversation: Any) -> dict[str, float]:
        """
        Assess relationship management in the given conversation.
        Returns a dictionary with relevant scores/metrics.
        """
        # TODO: Implement assessment logic
        return {"interpersonal_skills_score": 0.0}

    def evaluate(self, conversation: Any) -> dict[str, float]:
        """
        Run all EQ assessments and aggregate results.

        Args:
            conversation (Any): The conversation data to evaluate.

        Returns:
            dict[str, float]: Aggregated EQ scores for all domains.

        Example:
            >>> evaluator = EQEvaluator()
            >>> scores = evaluator.evaluate(conversation)
            >>> print(scores)
        """
        self.audit_log("evaluate_start", "started", {"evaluator": "EQEvaluator"})
        self.track_event("EQ evaluation started", {"conversation_id": id(conversation)})
        results = {}
        try:
            results.update(self.safe_execute(self.assess_emotional_awareness, conversation))
            results.update(self.safe_execute(self.assess_empathy_recognition, conversation))
            results.update(self.safe_execute(self.assess_emotional_regulation, conversation))
            results.update(self.safe_execute(self.assess_social_cognition, conversation))
            results.update(self.safe_execute(self.assess_interpersonal_skills, conversation))
            # Optionally, add overall EQ scoring/aggregation here
            self.audit_log("evaluate_end", "success", {"evaluator": "EQEvaluator"})
            self.track_event("EQ evaluation completed", {"conversation_id": id(conversation)})
            return results
        except Exception as e:
            self.audit_log("evaluate_end", "error", {"evaluator": "EQEvaluator", "error": str(e)})
            self.track_event(
                "EQ evaluation error", {"conversation_id": id(conversation), "error": str(e)}
            )
            raise
