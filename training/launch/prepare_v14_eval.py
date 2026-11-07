"""Prepare the paired-eval configs for the v14 release.

Resolves the v14 LoRA artifact URI from its W&B run and writes one config per
held-out suite for two comparisons:

* ``v14-vs-base`` -- the absolute reading (base model vs v14 adapter).
* ``v14-vs-v13``  -- the direct reading (v13 adapter vs v14 adapter).

The runner schema hard-codes exactly two arms named ``base`` and ``sft``, so the
direct comparison puts v13 in the ``base`` slot and v14 in the ``sft`` slot.

Run ids embed a digest of the compared artifact URIs, so re-running the same
artifact resumes cleanly while a new adapter gets a fresh identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import wandb

ENTITY = "wutang"
PROJECT = "pixelated-empathy-kan28"
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
V13_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-ecf35d5e5384:v1"
V14_RUN_NAME = "arc-sft-baseline-8ccb925b683e"
INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
LORA_ARTIFACT_TYPE = "lora"
API_KEY_ENV = "WANDB_API_KEY"

DECODING: dict[str, Any] = {
    "temperature": 0,
    "max_tokens": 2048,
    "stop": [],
    "chat_template_kwargs": {"enable_thinking": False},
}

# The runner refuses to enforce a priced hard cap without an input-token
# preflight, and the spend caps below are sized against exactly these values
# (the same ones the v13 suites used). Keep them in step.
MAX_INPUT_TOKENS = 4096
MAX_TOOL_ROUNDS = 4

# (suite label, case file, spend cap in USD) -- caps mirror the v13 configs.
SUITES: tuple[tuple[str, str, float], ...] = (
    ("capability-dev", "scenario-dev-v1.jsonl", 0.2),
    ("capability-sealed", "scenario-sealed-v1.jsonl", 0.2),
    ("crisis-dev", "scenario-crisis-dev-v1.jsonl", 0.5),
    ("crisis-sealed", "scenario-crisis-sealed-v1.jsonl", 0.5),
    ("literal-dev", "scenario-literal-dev-v1.jsonl", 0.3),
    ("literal-sealed", "scenario-literal-sealed-v1.jsonl", 0.3),
    ("contract-dev", "scenario-contract-dev-v1.jsonl", 0.2),
    ("contract-sealed", "scenario-contract-sealed-v1.jsonl", 0.2),
)

PAIRS: tuple[str, ...] = ("v14-vs-base", "v14-vs-v13")


def default_cases_dir() -> Path:
    """Locate the shared scenario suites in the docs submodule."""
    return Path(__file__).resolve().parents[3] / "docs" / "training" / "execution-2026-10-17"


def resolve_v14_uri(run_name: str = V14_RUN_NAME) -> str:
    """Return the newest lora artifact URI logged by the v14 training run."""
    run = wandb.Api().run(f"{ENTITY}/{PROJECT}/runs/{run_name}")
    if run.state != "finished":
        raise RuntimeError(f"run {run_name} is {run.state}; the artifact is only final once it finishes")
    lora = [artifact for artifact in run.logged_artifacts() if artifact.type == LORA_ARTIFACT_TYPE]
    if not lora:
        raise RuntimeError(f"run {run_name} logged no {LORA_ARTIFACT_TYPE} artifact")
    newest = max(lora, key=lambda artifact: int(artifact.version.lstrip("v")))
    # artifact.name is "collection:version" in current SDKs, so the version must
    # not be appended to the whole name or the URI ends up with ":v1:v1".
    collection = newest.name.split(":", 1)[0]
    return f"wandb-artifact:///{ENTITY}/{PROJECT}/{collection}:{newest.version}"


def _arm(arm_id: str, model: str, provider: str) -> dict[str, str]:
    return {
        "id": arm_id,
        "provider": provider,
        "base_url": INFERENCE_BASE_URL,
        "model": model,
        "project": f"{ENTITY}/{PROJECT}",
        "api_key_env": API_KEY_ENV,
    }


def build_config(suite: str, pair: str, v14_uri: str, hard_cap_usd: float) -> dict[str, Any]:
    """Build one paired-eval config for a suite/comparison pair."""
    if pair == "v14-vs-base":
        base_arm = _arm("base", BASE_MODEL, "wandb-inference")
    elif pair == "v14-vs-v13":
        base_arm = _arm("base", V13_URI, "wandb-serverless-training")
    else:
        raise ValueError(f"unknown comparison pair: {pair}")
    digest = hashlib.sha256(f"{suite}:{pair}:{v14_uri}".encode()).hexdigest()[:8]
    return {
        "schema_version": 1,
        "run_id": f"wandb-{suite}-{pair}-{digest}",
        "decoding": dict(DECODING),
        "pricing": {
            "input_usd_per_million_tokens": 0.25,
            "output_usd_per_million_tokens": 1.25,
            "hard_cap_usd": hard_cap_usd,
            "require_usage": True,
            "require_finish_reason": True,
        },
        "arms": [
            base_arm,
            _arm("sft", v14_uri, "wandb-serverless-training"),
        ],
    }


def count_cases(cases: Path) -> int:
    """Count the cases in a scenario JSONL file."""
    return sum(1 for line in cases.read_text(encoding="utf-8").splitlines() if line.strip())


def estimate_max_cost_usd(config: dict[str, Any], case_count: int) -> float:
    """Mirror the runner's worst-case spend estimate for one config.

    The runner raises if this exceeds ``pricing.hard_cap_usd``, so checking it
    here turns a cap/suite mismatch into a config-time failure instead of a
    surprise an hour into an eval run.
    """
    pricing = config["pricing"]
    per_call = (
        MAX_INPUT_TOKENS * pricing["input_usd_per_million_tokens"]
        + config["decoding"]["max_tokens"] * pricing["output_usd_per_million_tokens"]
    )
    calls_per_case = MAX_TOOL_ROUNDS + 1
    return case_count * len(config["arms"]) * calls_per_case * per_call / 1_000_000


def write_configs(
    cases_dir: Path,
    out_dir: Path,
    v14_uri: str,
) -> list[tuple[str, Path, Path, bool]]:
    """Write every suite/pair config; return (suite, cases, config, sealed) rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, Path, Path, bool]] = []
    for suite, case_file, hard_cap_usd in SUITES:
        cases = cases_dir / case_file
        if not cases.is_file():
            raise FileNotFoundError(f"missing case file for {suite}: {cases}")
        case_count = count_cases(cases)
        for pair in PAIRS:
            config = build_config(suite, pair, v14_uri, hard_cap_usd)
            estimate = estimate_max_cost_usd(config, case_count)
            if estimate > hard_cap_usd:
                raise ValueError(f"{suite}-{pair} worst case ${estimate:.4f} exceeds its ${hard_cap_usd:.2f} cap")
            path = out_dir / f"wandb-{suite}-{pair}.config.json"
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            written.append((f"{suite}-{pair}", cases, path, "-sealed" in suite))
    return written


def render_plan(v14_uri: str, written: list[tuple[str, Path, Path, bool]], results_root: Path) -> str:
    """Render the artifact line plus the exact runner command per config."""
    lines = [f"v14 artifact: {v14_uri}"]
    for label, cases, config, sealed in written:
        sealed_flag = " --allow-sealed" if sealed else ""
        lines.append(
            f"{label}\n  python training/run_wandb_paired_eval.py --cases {cases} "
            f"--config {config} --results-dir {results_root / label} "
            f"--max-input-tokens {MAX_INPUT_TOKENS} --max-tool-rounds {MAX_TOOL_ROUNDS}{sealed_flag}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v14-uri", default=None, help="skip W&B resolution and use this artifact URI")
    parser.add_argument("--run-name", default=V14_RUN_NAME)
    parser.add_argument("--cases-dir", type=Path, default=default_cases_dir())
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.cases_dir.parent / "execution-2026-11-06"
    v14_uri = args.v14_uri or resolve_v14_uri(args.run_name)
    written = write_configs(args.cases_dir, out_dir, v14_uri)
    sys.stdout.write(render_plan(v14_uri, written, out_dir / "results") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
