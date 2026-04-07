from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai.pipelines.orchestrator.orchestration.dataset_output_service import (
    DatasetOutputService,
)
from ai.pipelines.orchestrator.orchestration.standard_therapeutic_loader_service import (
    StandardTherapeuticLoaderService,
)


@dataclass
class _LoaderConfigStub:
    source_path: str | None
    source_paths: tuple[str, ...] = ()
    fallback_paths: tuple[str, ...] = ()
    max_samples: int | None = None


@dataclass
class _StatsStub:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    samples_by_source: dict[str, int] = field(default_factory=lambda: {"therapeutic": 2})
    samples_by_stage: dict[str, int] = field(default_factory=lambda: {"stage2_therapeutic_expertise": 2})
    stage_balance: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "stage2_therapeutic_expertise": {"target": 2, "available": 2, "actual": 2}
        }
    )
    split_counts: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class _OutputConfigStub:
    output_dir: str
    output_filename: str
    stage_distribution: dict[str, float] = field(
        default_factory=lambda: {"stage2_therapeutic_expertise": 1.0}
    )


def test_standard_therapeutic_loader_service_loads_and_normalizes_json(tmp_path: Path):
    dataset_dir = tmp_path / "standard"
    dataset_dir.mkdir()
    dataset_file = dataset_dir / "training_dataset.json"
    dataset_file.write_text(
        json.dumps(
            {
                "conversations": [
                    {"text": "Direct text example"},
                    {
                        "conversation": [
                            {"role": "user", "content": "I feel overwhelmed."},
                            {"role": "assistant", "content": "Let's slow down together."},
                        ]
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    stats = _StatsStub()
    service = StandardTherapeuticLoaderService(
        config=_LoaderConfigStub(source_path=str(dataset_dir)),
        stats=stats,
        cache_data=lambda path: None,
    )

    records = service.load()

    assert len(records) == 2
    assert records[0]["metadata"]["source"] == "standard_therapeutic"
    assert "User: I feel overwhelmed." in records[1]["text"]


def test_standard_therapeutic_loader_service_honors_max_samples_across_sources(tmp_path: Path):
    primary = tmp_path / "primary.jsonl";
    primary.write_text("\n".join([json.dumps({"prompt": "p1", "response": "r1"}), json.dumps({"prompt": "p2", "response": "r2"})]) + "\n", encoding="utf-8")
    secondary = tmp_path / "secondary.jsonl";
    secondary.write_text("\n".join([json.dumps({"prompt": "p3", "response": "r3"}), json.dumps({"prompt": "p4", "response": "r4"})]) + "\n", encoding="utf-8")

    stats = _StatsStub()
    service = StandardTherapeuticLoaderService(
        config=_LoaderConfigStub(
            source_path=str(primary),
            source_paths=(str(secondary),),
            max_samples=3,
        ),
        stats=stats,
        cache_data=lambda path: None,
    )

    records = service.load()

    assert len(records) == 3
    assert [record["text"] for record in records] == [
        "User: p1\nAssistant: r1",
        "User: p2\nAssistant: r2",
        "User: p3\nAssistant: r3",
    ]


def test_dataset_output_service_writes_dataset_and_split_artifacts(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    stats = _StatsStub()
    config = _OutputConfigStub(
        output_dir=str(tmp_path / "output"),
        output_filename="training_dataset.json",
    )
    service = DatasetOutputService(config=config, stats=stats)
    records = [
        {"text": "a", "metadata": {"stage": "stage2_therapeutic_expertise", "split": "train"}},
        {"text": "b", "metadata": {"stage": "stage2_therapeutic_expertise", "split": "val"}},
        {"text": "c", "metadata": {"stage": "stage2_therapeutic_expertise", "split": "test"}},
    ]
    stage_segments = {"stage2_therapeutic_expertise": records}

    dataset_path = service.save_dataset(records)
    service.write_stage_outputs(stage_segments)
    service.write_split_outputs(records)

    assert Path(dataset_path).exists()
    saved = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    assert saved["metadata"]["total_conversations"] == 3

    manifest_path = Path("ai/training_data_consolidated/final/MASTER_STAGE_MANIFEST.json")
    assert manifest_path.exists()
    split_root = Path("ai/training_data_consolidated/final/splits")
    assert (split_root / "train.jsonl").exists()
    assert stats.split_counts["aggregate"]["train"] == 1
