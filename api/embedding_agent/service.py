    def _compute_similarity(
        self, item: Any, query_embedding: List[float], query_np: Any
    ) -> Optional[float]:
        # ⚡ Bolt: Cache normalized numpy arrays safely to avoid regenerating on query
        # Normalization awareness: Cache results based on current config to avoid stale vectors
        cached_tuple = getattr(item, "_cached_np_embedding", None)
        item_np = None
        
        if cached_tuple is not None:
            # cached_tuple format: (numpy_array, was_normalized)
            cached_np, was_normalized = cached_tuple
            if was_normalized == self.config.normalize_embeddings:
                item_np = cached_np

        if item_np is None:
            item_embedding = getattr(item, "embedding", None)
            if item_embedding is None:
                return None
            
            if NUMPY_AVAILABLE and query_np is not None:
                item_np = np.array(item_embedding)
                if self.config.normalize_embeddings:
                    norm = np.linalg.norm(item_np)
                    if norm > 0:
                        item_np = item_np / norm
                
                # Thread-safe cache population
                with self._lock:
                    try:
                        setattr(item, "_cached_np_embedding", (item_np, self.config.normalize_embeddings))
                    except AttributeError as e:
                        # Log caching failures for visibility (Review suggestion)
                        logger.debug(f"Skipped caching for {type(item).__name__}: {str(e)}")

        # Numpy path (Optimized)
        if NUMPY_AVAILABLE and query_np is not None and item_np is not None:
            return float(np.dot(query_np, item_np))

        # Fallback path (Standard)
        item_embedding = getattr(item, "embedding", None)
        if item_embedding is None:
            return None
            
        # Use strict=True to detect dimension mismatch early (Review suggestion)
        try:
            dot_product = sum(a * b for a, b in zip(query_embedding, item_embedding, strict=True))
        except ValueError as e:
            logger.error(f"Embedding dimension mismatch: {str(e)}")
            return 0.0
            
        norm_q = sum(x**2 for x in query_embedding) ** 0.5
        norm_i = sum(x**2 for x in item_embedding) ** 0.5
        return dot_product / (norm_q * norm_i) if norm_q * norm_i > 0 else 0.0
