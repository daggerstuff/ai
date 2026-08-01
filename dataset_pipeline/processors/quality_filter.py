import hashlib
import re
from collections import Counter

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
    "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves"
}

JUNK_ARTIFACTS = [
    "[inst]", "[/inst]", "<<sys>>", "<</sys>>", "<s>", "</s>",
    "{'source'", '{"source"', "<|im_start|>", "<|im_end|>",
    "http://", "https://", "www.", "traceback (most recent call last):",
    "import pandas as pd", "import numpy as np", "error:", "syntaxerror:",
    "undefined", "null", "[object object]", "<div", "<span", "</html"
]


class QualityFilter:
    """Enhanced quality filter with strict deduplication, content quality, and anti-repetition rules."""

    def __init__(self):
        self.seen_hashes: set[str] = set()
        self.seen_fingerprints: set[str] = set()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for hash comparison (lowercase, collapsed whitespace)."""
        return re.sub(r"\s+", " ", str(text).lower().strip())

    def _has_ngram_repetition(self, text: str) -> bool:
        """Detect if any 2-6 word phrase repeats consecutively or excessively."""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 5:
            return False

        # 1. Consecutive phrase repeats (2-word phrases repeated 3+ times, or 3-6 word phrases repeated 2+ times)
        for n in range(2, 7):
            min_repeats = 3 if n == 2 else 2
            for i in range(len(words) - n * min_repeats + 1):
                chunks = [words[i + k * n : i + (k + 1) * n] for k in range(min_repeats)]
                if all(c == chunks[0] for c in chunks):
                    return True

        # 2. Non-consecutive 3-gram high frequency check (> 3 occurrences of non-stop 3-gram in text)
        if len(words) >= 15:
            three_grams = [
                tuple(words[i : i + 3])
                for i in range(len(words) - 2)
                if not all(w in STOP_WORDS for w in words[i : i + 3])
            ]
            counts = Counter(three_grams)
            if any(cnt >= 4 for cnt in counts.values()):
                return True

        # 3. Character sequence repeats (e.g. "aaaaaa", "!!!!!!")
        if re.search(r"(.)\1{5,}", text):
            return True

        return False

    def _has_high_content_word_density(self, text: str, max_density: float = 0.18, min_words: int = 10) -> bool:
        """Detect if any non-stop word accounts for > max_density of total words."""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < min_words:
            return False
        content_words = [w for w in words if w not in STOP_WORDS]
        if not content_words:
            return False
        top_count = Counter(content_words).most_common(1)[0][1]
        return (top_count / len(words)) > max_density

    def _has_low_unique_word_ratio(self, text: str, min_ratio: float = 0.40, min_words: int = 20) -> bool:
        """Detect if unique word ratio is suspiciously low (indicating repetitive filler)."""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < min_words:
            return False
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio < min_ratio

    def _contains_junk_artifacts(self, text: str) -> bool:
        """Check for meta-artifacts, code snippets, prompt templates, or HTML tags."""
        text_lower = text.lower()
        return any(artifact in text_lower for artifact in JUNK_ARTIFACTS)

    def passes_filter(self, chatml_record: dict) -> bool:
        """Returns True if record passes strict quality, length, deduplication, and anti-repetition criteria."""
        if not isinstance(chatml_record, dict):
            return False

        messages = chatml_record.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            return False

        # 1. Role structure validation (ignoring system prompt at index 0)
        for i in range(len(messages) - 1):
            role1 = messages[i].get("role") if isinstance(messages[i], dict) else None
            role2 = messages[i + 1].get("role") if isinstance(messages[i + 1], dict) else None
            if i == 0 and role1 == "system":
                continue
            if role1 == role2:
                return False

        # 2. Strict Length & Substance Check
        user_msg = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
        asst_msg = next((m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"), None)

        if not user_msg or not asst_msg:
            return False

        user_content = str(user_msg.get("content", "")).strip()
        asst_content = str(asst_msg.get("content", "")).strip()

        # Minimum length: user >= 10 chars, assistant >= 25 chars
        if len(user_content) < 10 or len(asst_content) < 25:
            return False

        # 3. Artifact & Junk Text Checks across all messages
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            content = str(msg.get("content", "")).strip()
            if self._contains_junk_artifacts(content):
                return False

        # 4. Anti-Repetition, Word Density & Unique Ratio Checks on Assistant Turn
        if self._has_ngram_repetition(asst_content):
            return False
        if self._has_high_content_word_density(asst_content):
            return False
        if self._has_low_unique_word_ratio(asst_content):
            return False

        # 5. Deterministic Exact Deduplication via SHA-256
        concat_content = "".join(
            [f"{m.get('role', '')}:{self._normalize_text(m.get('content', ''))}" for m in messages if isinstance(m, dict)]
        )
        content_hash = hashlib.sha256(concat_content.encode("utf-8")).hexdigest()
        if content_hash in self.seen_hashes:
            return False

        # 6. Strict Normalized Prompt + Response Fingerprint Deduplication (first 120 prompt + first 100 response chars)
        norm_user = self._normalize_text(user_content)[:120]
        norm_asst = self._normalize_text(asst_content)[:100]

        if norm_user and norm_asst:
            fingerprint = hashlib.md5(f"{norm_user}||{norm_asst}".encode("utf-8")).hexdigest()
            if fingerprint in self.seen_fingerprints:
                return False
            self.seen_fingerprints.add(fingerprint)

        self.seen_hashes.add(content_hash)
        return True



