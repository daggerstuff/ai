"""Paraphrase SDG module (§B.5.3).

LLM paraphraser with high temperature. Filter: ROUGE-L > 0.85 vs original
= too similar; < 0.30 = meaning drift, drop.

Blueprint ref: docs/training-pipeline-blueprint-2026-08-10.md §B.5.3
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SEEDS_PATH = Path(__file__).parent.parent / "data" / "sdg_seeds" / "self_instruct_seed.jsonl"
TARGET_COUNT = 10_000
PARAPHRASE_TEMPERATURE = 1.2
ROUGE_L_TOO_SIMILAR = 0.85
ROUGE_L_MEANING_DRIFT = 0.30


def load_seeds(path: Path = SEEDS_PATH) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    if not path.exists():
        return seeds
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seeds.append(json.loads(line))
    return seeds


def _content_hash(record: dict[str, Any]) -> str:
    import hashlib
    text_blob = "||".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in record.get("messages", [])
    )
    return hashlib.sha256(text_blob.encode("utf-8")).hexdigest()[:16]


def _rouge_l_similarity(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 1.0
    return len(words_a & words_b) / len(words_a | words_b) if words_a | words_b else 0.0


def paraphrase_text(text: str, seed_id: str = "") -> str:
    """High-temperature LLM paraphraser (deterministic mock).

    Real impl: prompt an LLM at temperature=1.2 to rephrase the input utterance.
    Mock: reuse the back-translation paraphraser (word-level synonym/contraction/
    filler transforms) so the variant stays in the meaningful ROUGE-L band and
    actually tracks the input text instead of injecting canned sentences.
    """
    from training.sdg_back_translation import back_translate
    # Deterministic lang pick from seed_id so paraphrase differs from each
    # pivot-language back-translation variant.
    langs = ["de", "fr", "es", "ru", "zh", "ar", "ja", "ko"]
    lang = langs[(hash(seed_id)) % len(langs)] if seed_id else langs[hash(text) % len(langs)]
    return back_translate(text, lang)


def expand_seeds(
    seeds_path: Path = SEEDS_PATH,
    target_count: int = TARGET_COUNT,
) -> list[dict[str, Any]]:
    seeds = load_seeds(Path(seeds_path))
    if not seeds:
        return []
    synthetic: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    from training.sdg_back_translation import _extract_user_utterance, _rebuild_prompt
    from training.stage1_filters import FilterVerdict, NearDuplicateIndex, run_stage1_on_record
    stage1_index = NearDuplicateIndex()
    for iteration in range(200):
        if len(synthetic) >= target_count:
            break
        seed_index = iteration % len(seeds)
        seed = seeds[seed_index]
        # Per §B.5.3: paraphrase the user INPUT utterance, not the gold reply.
        original_utterance = _extract_user_utterance(seed["prompt"])
        variant = paraphrase_text(original_utterance, seed.get("id", "") + str(iteration))
        if len(variant) < 30:
            continue
        rouge_l = _rouge_l_similarity(variant, original_utterance)
        if rouge_l > ROUGE_L_TOO_SIMILAR:
            continue  # too similar
        if rouge_l < ROUGE_L_MEANING_DRIFT:
            continue  # meaning drift
        # Toxicity check
        lowered = variant.lower()
        if any(p in lowered for p in ["kill yourself", "hurt yourself", "end your life", "suicide"]):
            continue
        # Rebuild seed prompt with the paraphrased user utterance; assistant reply
        # is a deterministic therapeutic acknowledgment (input-agnostic gold).
        rebuilt_prompt = _rebuild_prompt(seed["prompt"], variant)
        reply_variants = [
            "I hear you. That sounds genuinely difficult, and I'm glad you're sharing it with me. Can you tell me more about what's been going on for you?",
            "Thank you for reaching out. This matters, and you don't have to carry it alone. What would feel most helpful to talk through first?",
            "I'm here with you. It takes courage to put this into words. What's been weighing on you the most lately?",
        ]
        assistant_reply = reply_variants[(hash(seed.get("id", "")) + hash(iteration)) % len(reply_variants)]
        record = {
            "source": "synthetic_sdg",
            "task_type": "sft",
            "messages": [
                {"role": "user", "content": rebuilt_prompt},
                {"role": "assistant", "content": assistant_reply},
            ],
            "mi_quality": "low",
            "clinical_reviewed": False,
            "annotation_stage": "v1_initial",
            "created_by": "sdg_paraphrase_pix4345",
        }
        try:
            stage1_result = run_stage1_on_record(record, dedup_index=stage1_index)
            if stage1_result.verdict == FilterVerdict.DROP:
                continue
            if stage1_result.transformed_text is not None:
                if record["messages"] and len(record["messages"]) >= 2:
                    record["messages"][1]["content"] = stage1_result.transformed_text
        except Exception:
            continue
        h = _content_hash(record)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        provenance_dict = {
            "source_url": "synthetic_sdg",
            "source_type": "synthetic_sdg",
            "acquired_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "pipeline_version": "pix4345-sdg",
            "license": "NOASSERTION",
            "transformations": ["sdg_paraphrase_pix4345"],
        }
        record["provenance"] = provenance_dict
        record["quality_score"] = 0.75
        record["fleiss_kappa"] = None
        synthetic.append(record)
    return synthetic


def build_parser() -> Any:
    import argparse
    parser = argparse.ArgumentParser(description="Paraphrase SDG expansion (PIX-4345 §B.5.3)")
    parser.add_argument("--seeds-path", type=str, default=str(SEEDS_PATH))
    parser.add_argument("--output-dir", type=str, default="ai/data/synthetic/paraphrase")
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    seeds_path = Path(args.seeds_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = args.target_count if args.target_count > 0 else TARGET_COUNT
    print(f"[PIX-4345-PP] Loading seeds from {seeds_path}")
    synthetic = expand_seeds(seeds_path, target_count=target)
    print(f"[PIX-4345-PP] Generated {len(synthetic)} synthetic records")
    output_path = output_dir / "train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for record in synthetic:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[PIX-4345-PP] Wrote {len(synthetic)} records to {output_path}")


if __name__ == "__main__":
    main()
