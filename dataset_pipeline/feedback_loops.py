    def identify_anti_patterns(self) -> List[Dict[str, Any]]:
        """
        Analyze the memory buffer to find common features in highly requested/poorly rated prompts.
        This provides structured directives to the generation layer to stop
        producing conversational dead-ends.

        PERFORMANCE NOTE: Reduces Python iteration overhead by consolidating passes.
        Maintains O(N*M) algorithmic complexity but avoids repeated list traversal.
        """
        logger.info("Analyzing feedback buffer for anti-patterns...")

        anti_patterns = []
        failure_contexts = [
            fb.context
            for fb in self.memory_buffer
            if fb.rating < self.config["confidence_threshold"]
        ]

        if not failure_contexts:
            return []

        try:
            # ⚡ Bolt: Consolidated pass over failure contexts (Review suggestion)
            dummy_keywords = ["toxic positivity", "abrupt ending", "unhelpful generic"]
            keyword_counts = {kw: 0 for kw in dummy_keywords}

            for context in failure_contexts:
                # Check all keywords in one pass of the context string
                for keyword in dummy_keywords:
                    if keyword in context:
                        keyword_counts[keyword] += 1

            for keyword, frequency in keyword_counts.items():
                if frequency > 5:
                    anti_patterns.append(
                        {
                            "pattern": keyword,
                            "frequency": frequency,
                            "severity": "high" if frequency > 20 else "medium",
                        }
                    )
        except Exception as e:
            logger.error(f"Error during anti-pattern extraction: {e}")

        return anti_patterns
