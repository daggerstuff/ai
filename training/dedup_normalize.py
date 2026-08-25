#!/usr/bin/env python3
"""Deduplication and ChatML normalization pipeline.

Exact dedup via SHA-256 content hash, near-dedup via Jaccard token similarity
or MinHash/LSH banding (PIX-4242) for O(1) bucket lookup at scale,
edge-case preservation, ChatML boundary verification, and sharded JSONL output.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("dedup_normalize")

_INST_BOUNDARY = re.compile(r"\[/INST\]")

# datasketch is optional — pure-python MinHash/LSH fallback is used when missing.
try:
    from datasketch import MinHash as _DMinHash, MinHashLSH as _DMinHashLSH

    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False


@dataclasses.dataclass
class ProcessingContext:
    seen_hashes: set[str]
    edge_case_hashes: set[str]
    token_sets: list[tuple[frozenset[str], str]]
    lsh_index: _MinHashIndex | None = None  # None = Jaccard window mode
    simhash_index: SimHashIndex | None = None  # None = simhash disabled


@dataclasses.dataclass
class DedupStats:
    kept: list[dict]
    exact_dupes: int
    near_dupes: int
    chatml_failures: int
    reformatted: int
    total_read: int


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_set(text: str) -> frozenset[str]:
    return frozenset(text.lower().split())


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


_HASH_PERM_CACHE: dict[tuple[int, str], int] = {}


def _hash_token_perm(token: str, perm_idx: int, seed: int = 42) -> int:
    key = (seed + perm_idx, token)
    cached = _HASH_PERM_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256(f"{seed + perm_idx}:{token}".encode()).digest()
    val = int.from_bytes(h[:8], "little")
    _HASH_PERM_CACHE[key] = val
    return val


def _minhash_signature(text: str, num_perm: int = 128, seed: int = 42) -> list[int]:
    """Pure-python MinHash signature.

    Each permutation tracks the minimum hash across tokens. Empty text
    returns a zero signature (matches itself, so dedup keeps first only).
    """
    tokens = text.lower().split()
    if not tokens:
        return [0] * num_perm
    sig = [float("inf")] * num_perm
    for token in tokens:
        for i in range(num_perm):
            h = _hash_token_perm(token, i, seed)
            if h < sig[i]:
                sig[i] = h
    return [0 if s == float("inf") else int(s) for s in sig]


def _minhash_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures.

    Empty input = identical (treat as 1.0 like the token-set Jaccard path).
    """
    if not sig_a and not sig_b:
        return 1.0
    if not sig_a or not sig_b:
        return 0.0
    if len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


_SIMHASH_BITS = 64
_SIMHASH_BANDS = 4  # 64 bits / 16-bit bands: Hamming <= 3 => at least one band identical


def _simhash_signature(text: str, bits: int = _SIMHASH_BITS) -> int:
    """64-bit SimHash fingerprint (PIX-4242 cross-source dedup).

    Sums each token's 64-bit hash per bit position (+1 set, -1 unset); the
    fingerprint bit is 1 where the sum is positive. Empty text yields 0 so it
    matches itself (Hamming 0), consistent with the other dedup paths.
    """
    tokens = text.lower().split()
    vector = [0] * bits
    for token in tokens:
        h = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "little")
        for i in range(bits):
            vector[i] += 1 if (h >> i) & 1 else -1
    sig = 0
    for i in range(bits):
        if vector[i] > 0:
            sig |= 1 << i
    return sig


def _hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two SimHash fingerprints."""
    return (a ^ b).bit_count()


class SimHashIndex:
    """Bucketed index over 64-bit SimHash fingerprints.

    Splits the fingerprint into ``bands`` equal-width chunks. By the pigeonhole
    principle, two fingerprints within Hamming distance ``<= 3`` share at least
    one identical 16-bit chunk (for 64 bits / 4 bands), so candidates are found
    by exact chunk lookup and confirmed with an exact Hamming comparison.
    """

    def __init__(
        self,
        hamming_threshold: int = 3,
        bits: int = _SIMHASH_BITS,
        bands: int = _SIMHASH_BANDS,
    ) -> None:
        if bits % bands != 0:
            raise ValueError(f"bits ({bits}) must be divisible by bands ({bands})")
        self.hamming_threshold = hamming_threshold
        self.bits = bits
        self.bands = bands
        self.band_bits = bits // bands
        self.signatures: dict[str, int] = {}
        self.buckets: list[dict[int, list[str]]] = [
            defaultdict(list) for _ in range(bands)
        ]

    def _band(self, signature: int, band: int) -> int:
        shift = band * self.band_bits
        mask = (1 << self.band_bits) - 1
        return (signature >> shift) & mask

    def add(self, text_hash: str, signature: int) -> None:
        self.signatures[text_hash] = signature
        for b in range(self.bands):
            self.buckets[b][self._band(signature, b)].append(text_hash)

    def query_candidates(self, signature: int) -> set[str]:
        candidates: set[str] = set()
        for b in range(self.bands):
            for h in self.buckets[b].get(self._band(signature, b), ()):
                candidates.add(h)
        return candidates

    def signature_for(self, text_hash: str) -> int | None:
        return self.signatures.get(text_hash)

    def is_near_duplicate(self, signature: int, text_hash: str) -> bool:
        """True if ``signature`` is within Hamming threshold of any indexed sig."""
        for cand_hash in self.query_candidates(signature):
            if cand_hash == text_hash:
                continue
            cand_sig = self.signature_for(cand_hash)
            if cand_sig is None:
                continue
            if _hamming_distance(signature, cand_sig) <= self.hamming_threshold:
                return True
        return False


class _MinHashIndex:
    """LSH banding index over MinHash signatures.

    Configurable via num_perm / bands / rows (num_perm must equal bands * rows).
    Buckets records by band signatures so query returns only same-band candidates;
    exact signature Jaccard then filters against the provided threshold.
    Falls back to pure-python if datasketch is unavailable — the datasketch
    branch is unused by default to avoid a hard dependency.
    """

    def __init__(
        self,
        num_perm: int = 128,
        bands: int = 16,
        rows: int = 8,
        jaccard_threshold: float = 0.85,
        use_datasketch: bool = False,
    ) -> None:
        if num_perm != bands * rows:
            raise ValueError(
                f"num_perm ({num_perm}) must equal bands * rows ({bands * rows})",
            )
        self.num_perm = num_perm
        self.bands = bands
        self.rows = rows
        self.threshold = jaccard_threshold
        self.use_datasketch = use_datasketch and _HAS_DATASKETCH
        self.signatures: list[tuple[list[int], str]] = []
        self.buckets: list[dict[tuple[int, ...], list[str]]] = [defaultdict(list) for _ in range(bands)]
        self.hash_to_sig: dict[str, list[int]] = {}
        if self.use_datasketch:
            self._d_lsh = _DMinHashLSH(
                num_perm=num_perm,
                params=((bands, rows),),
            )

    def add(self, text_hash: str, signature: list[int]) -> None:
        self.signatures.append((signature, text_hash))
        self.hash_to_sig[text_hash] = signature
        for b in range(self.bands):
            band = tuple(signature[b * self.rows : (b + 1) * self.rows])
            self.buckets[b][band].append(text_hash)
        if self.use_datasketch:
            mh = _DMinHash(num_perm=self.num_perm)
            for t in signature:
                mh.update(t.to_bytes(8, "little"))
            self._d_lsh.insert(text_hash, mh)

    def query_candidates(self, signature: list[int]) -> set[str]:
        if self.use_datasketch:
            mh = _DMinHash(num_perm=self.num_perm)
            for t in signature:
                mh.update(t.to_bytes(8, "little"))
            return set(self._d_lsh.query(mh))
        candidates: set[str] = set()
        for b in range(self.bands):
            band = tuple(signature[b * self.rows : (b + 1) * self.rows])
            for h in self.buckets[b].get(band, ()):
                candidates.add(h)
        return candidates

    def signature_for(self, text_hash: str) -> list[int] | None:
        return self.hash_to_sig.get(text_hash)

    def is_near_duplicate(self, new_sig: list[int], text_hash: str) -> bool:
        """Check whether new_sig collides with any existing signature above threshold."""
        candidates = self.query_candidates(new_sig)
        for cand_hash in candidates:
            if cand_hash == text_hash:
                continue
            cand_sig = self.signature_for(cand_hash)
            if cand_sig is None:
                continue
            if _minhash_jaccard(new_sig, cand_sig) > self.threshold:
                return True
        return False


def _extract_text(record: dict) -> str:
    messages = record.get("messages", [])
    if messages:
        return " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
    if record.get("prompt") and record.get("chosen") and record.get("rejected"):
        return record["prompt"] + " " + record["chosen"] + " " + record["rejected"]
    return record.get("text", "") or record.get("instruction", "") + " " + record.get("output", "")


def _is_edge_case(record: dict) -> bool:
    return record.get("is_training_edge_case", False) is True


def _verify_chatml_boundary(record: dict) -> bool:
    text = _extract_text(record)
    if "[/INST]" in text:
        return bool(_INST_BOUNDARY.search(text))
    if "messages" in record:
        return True
    return True


def _attempt_reformat(record: dict) -> dict | None:
    """Try to reformat a record to have proper ChatML boundaries."""
    text = _extract_text(record)
    if not text:
        return None

    if "[/INST]" not in text and "messages" not in record:
        instruction = record.get("instruction", "")
        output = record.get("output", "")
        if instruction and output:
            record = {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ],
                "metadata": record.get("metadata", {}),
            }
            if _is_edge_case(record) if "is_training_edge_case" in record else False:
                record["is_training_edge_case"] = True
            return record
    return None


def process_file(
    input_path: Path,
    jaccard_threshold: float,
    rejection_log: list[dict],
    ctx: ProcessingContext,
    near_dedup_window: int = 2000,
) -> DedupStats:
    """Process one JSONL file.

    Returns DedupStats.
    """
    kept: list[dict] = []
    exact_dupes = 0
    near_dupes = 0
    chatml_failures = 0
    reformatted = 0
    total_read = 0

    try:
        with open(input_path, encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                total_read += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in %s line %d", input_path, line_no)
                    continue

                text = _extract_text(record)
                text_hash = _content_hash(text)

                if _is_edge_case(record):
                    ctx.edge_case_hashes.add(text_hash)
                    if not _verify_chatml_boundary(record):
                        reformatted_rec = _attempt_reformat(record)
                        if reformatted_rec:
                            record = reformatted_rec
                            reformatted += 1
                        else:
                            chatml_failures += 1
                            rejection_log.append(
                                {
                                    "file": str(input_path),
                                    "line": line_no,
                                    "reason": "ChatML boundary failure (edge case)",
                                }
                            )
                            continue
                    kept.append(record)
                    continue

                if text_hash in ctx.seen_hashes or text_hash in ctx.edge_case_hashes:
                    exact_dupes += 1
                    continue

                tokens = _token_set(text)
                is_near_dup = False
                if ctx.simhash_index is not None:
                    sig = _simhash_signature(text)
                    is_near_dup = ctx.simhash_index.is_near_duplicate(sig, text_hash)
                    if not is_near_dup:
                        ctx.simhash_index.add(text_hash, sig)
                elif ctx.lsh_index is not None:
                    sig = _minhash_signature(text, ctx.lsh_index.num_perm)
                    is_near_dup = ctx.lsh_index.is_near_duplicate(sig, text_hash)
                    if not is_near_dup:
                        ctx.lsh_index.add(text_hash, sig)
                else:
                    compare_window = ctx.token_sets[-near_dedup_window:] if near_dedup_window else ctx.token_sets
                    for existing_tokens, existing_hash in compare_window:
                        if existing_hash == text_hash:
                            continue
                        if _jaccard_similarity(tokens, existing_tokens) > jaccard_threshold:
                            is_near_dup = True
                            break

                if is_near_dup:
                    near_dupes += 1
                    continue

                if not _verify_chatml_boundary(record):
                    reformatted_rec = _attempt_reformat(record)
                    if reformatted_rec:
                        record = reformatted_rec
                        reformatted += 1
                    else:
                        chatml_failures += 1
                        rejection_log.append(
                            {
                                "file": str(input_path),
                                "line": line_no,
                                "reason": "ChatML boundary failure",
                            }
                        )
                        continue

                ctx.seen_hashes.add(text_hash)
                ctx.token_sets.append((tokens, text_hash))
                kept.append(record)

    except OSError as exc:
        logger.warning("Cannot read %s: %s", input_path, exc)

    return DedupStats(
        kept=kept,
        exact_dupes=exact_dupes,
        near_dupes=near_dupes,
        chatml_failures=chatml_failures,
        reformatted=reformatted,
        total_read=total_read,
    )


def run_dedup(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jaccard_threshold = args.jaccard_threshold
    shard_size = args.shard_size

    lsh_index: _MinHashIndex | None = None
    simhash_index: SimHashIndex | None = None
    if args.semantic_dedup == "lsh":
        lsh_index = _MinHashIndex(
            num_perm=args.num_perm,
            bands=args.lsh_bands,
            rows=args.lsh_rows,
            jaccard_threshold=jaccard_threshold,
            use_datasketch=args.use_datasketch,
        )
        logger.info(
            "LSH enabled: num_perm=%d bands=%d rows=%d datasketch=%s",
            args.num_perm,
            args.lsh_bands,
            args.lsh_rows,
            lsh_index.use_datasketch,
        )
    elif args.semantic_dedup == "simhash":
        simhash_index = SimHashIndex(hamming_threshold=args.simhash_hamming_threshold)
        logger.info(
            "SimHash enabled: bits=%d bands=%d hamming_threshold=%d",
            simhash_index.bits,
            simhash_index.bands,
            simhash_index.hamming_threshold,
        )

    ctx = ProcessingContext(
        seen_hashes=set(),
        edge_case_hashes=set(),
        token_sets=[],
        lsh_index=lsh_index,
        simhash_index=simhash_index,
    )
    rejection_log: list[dict] = []

    all_kept: list[dict] = []
    total_in = 0
    total_exact_dup = 0
    total_near_dup = 0
    total_chatml_fail = 0
    total_reformatted = 0
    total_edge_preserved = 0

    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.warning("Input directory not found: %s", input_path)
            continue
        for jsonl_file in sorted(input_path.rglob("*.jsonl")):
            logger.info(f"Processing {jsonl_file.name}...")
            stats = process_file(
                jsonl_file,
                jaccard_threshold,
                rejection_log,
                ctx,
                args.near_dedup_window,
            )
            logger.info(
                f"  {jsonl_file.name}: {stats.total_read} read, {len(stats.kept)} kept, {stats.exact_dupes} exact, {stats.near_dupes} near dup"
            )
            all_kept.extend(stats.kept)
            total_in += stats.total_read
            total_exact_dup += stats.exact_dupes
            total_near_dup += stats.near_dupes
            total_chatml_fail += stats.chatml_failures
            total_reformatted += stats.reformatted
            total_edge_preserved += sum(1 for r in stats.kept if _is_edge_case(r))

    shard_count = 0
    if shard_size > 0:
        for i in range(0, len(all_kept), shard_size):
            shard = all_kept[i : i + shard_size]
            shard_path = output_dir / f"shard_{shard_count:04d}.jsonl"
            with open(shard_path, "w", encoding="utf-8") as f:
                for record in shard:
                    f.write(json.dumps(record) + "\n")
            shard_count += 1
    elif all_kept:
        shard_path = output_dir / "shard_0000.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for record in all_kept:
                f.write(json.dumps(record) + "\n")
        shard_count = 1

    if rejection_log:
        rejection_path = output_dir / args.rejection_log
        with open(rejection_path, "w", encoding="utf-8") as f:
            for entry in rejection_log:
                f.write(json.dumps(entry) + "\n")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dirs": args.input_dirs,
        "total_samples_in": total_in,
        "exact_duplicates": total_exact_dup,
        "near_duplicates": total_near_dup,
        "chatml_failures": total_chatml_fail,
        "reformatted": total_reformatted,
        "edge_cases_preserved": total_edge_preserved,
        "total_samples_out": len(all_kept),
        "shard_count": shard_count,
        "jaccard_threshold": jaccard_threshold,
        "shard_size": shard_size,
        "semantic_dedup": args.semantic_dedup,
        "lsh_config": {
            "num_perm": args.num_perm,
            "bands": args.lsh_bands,
            "rows": args.lsh_rows,
            "datasketch_used": bool(lsh_index and lsh_index.use_datasketch),
        }
        if lsh_index is not None
        else None,
        "simhash_config": {
            "bits": simhash_index.bits,
            "bands": simhash_index.bands,
            "hamming_threshold": simhash_index.hamming_threshold,
        }
        if simhash_index is not None
        else None,
    }
    report_path = output_dir / "normalization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    logger.info(
        "Dedup complete: %d in → %d out (%d exact dup, %d near dup, %d ChatML fail, %d edge preserved)",
        total_in,
        len(all_kept),
        total_exact_dup,
        total_near_dup,
        total_chatml_fail,
        total_edge_preserved,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deduplicate and normalize training data.",
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="Directories containing JSONL files to deduplicate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for sharded JSONL and reports.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistral-nemo",
        help="Model name for ChatML template verification.",
    )
    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for near-dedup.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="Maximum records per output shard.",
    )
    parser.add_argument(
        "--rejection_log",
        type=str,
        default="rejection_log.jsonl",
        help="Filename for ChatML rejection log.",
    )
    parser.add_argument(
        "--near_dedup_window",
        type=int,
        default=2000,
        help="Max prior token sets to compare for near-dedup (limits O(n^2) to O(n*window)).",
    )
    parser.add_argument(
        "--semantic_dedup",
        choices=["none", "lsh", "simhash"],
        default="none",
        help="Near-dedup strategy: 'none' uses Jaccard window, 'lsh' uses MinHash/LSH banding, 'simhash' uses 64-bit SimHash + Hamming.",
    )
    parser.add_argument(
        "--num_perm",
        type=int,
        default=128,
        help="MinHash permutations for LSH mode (must equal lsh_bands * lsh_rows).",
    )
    parser.add_argument(
        "--lsh_bands",
        type=int,
        default=16,
        help="LSH band count for semantic dedup.",
    )
    parser.add_argument(
        "--lsh_rows",
        type=int,
        default=8,
        help="LSH rows per band for semantic dedup.",
    )
    parser.add_argument(
        "--use_datasketch",
        action="store_true",
        help="Use datasketch MinHash/LSH if installed (otherwise pure-python fallback).",
    )
    parser.add_argument(
        "--simhash_hamming_threshold",
        type=int,
        default=3,
        help="Hamming distance threshold for SimHash dedup (near-duplicate if <= this).",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_dedup(args)


if __name__ == "__main__":
    main()
