    def evaluate_empathy(self, conversation: List[Dict[str, str]]) -> float:
        """
        Evaluate conversational empathy using keyword-based heuristics.
        Checks for validating/empathetic keywords in assistant responses.
        In production, this would use a fast local model or heuristic NLP.
        """
        empathy_keywords = [
