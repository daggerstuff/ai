from __future__ import annotations

import json
from pathlib import Path

from ai.tools.utilities.youtube_curation import (
    YouTubeRecord,
    curate_youtube_output,
    decide_record_action,
)

EXPECTED_INCLUDED_RECORDS = 2
EXPECTED_EXCLUDED_RECORDS = 2


def _record(*, channel: str, title: str) -> YouTubeRecord:
    return {
        "instruction": f"Use this real transcript excerpt from {channel} as therapeutic source material.",
        "language": "en",
        "output": "example output",
        "source_channel": channel,
        "provenance": {
            "metadata": {
                "channel": channel,
                "transcript_file": title,
            }
        },
    }


def test_decide_record_action_includes_trusted_therapeutic_channel() -> None:
    decision = decide_record_action(_record(channel="Tim Fletcher", title="Any Topic.txt"))

    assert decision.include is True
    assert decision.reason == "trusted_channel"


def test_decide_record_action_includes_mixed_channel_with_therapeutic_title() -> None:
    decision = decide_record_action(
        _record(
            channel="The Diary Of A CEO",
            title="The Body Trauma Expert: This Eye Movement Trick Can Fix Your Trauma!.txt",
        )
    )

    assert decision.include is True
    assert decision.reason == "strict_channel_authority_signal"


def test_decide_record_action_excludes_mixed_channel_without_therapeutic_signal() -> None:
    decision = decide_record_action(
        _record(
            channel="The Diary Of A CEO",
            title="A Founder Explains How To Scale A Startup.txt",
        )
    )

    assert decision.include is False
    assert decision.reason == "non_therapeutic_title"


def test_decide_record_action_excludes_strict_mixed_channel_without_authority_signal() -> None:
    decision = decide_record_action(
        _record(
            channel="The Diary Of A CEO",
            title="Former FBI Agent: If They Do This Please RUN! Narcissists Favourite Trick To Control You!.txt",
        )
    )

    assert decision.include is False
    assert decision.reason == "strict_channel_needs_authority_signal"


def test_decide_record_action_excludes_low_confidence_title_on_strict_channel() -> None:
    decision = decide_record_action(
        _record(
            channel="Mel Robbins",
            title="#1 Neurosurgeon: How to Manifest Anything You Want & Unlock the Unlimited Power of Your Mind.txt",
        )
    )

    assert decision.include is False
    assert decision.reason == "strict_channel_low_confidence_title"


def test_decide_record_action_includes_strict_mixed_channel_with_named_expert_signal() -> None:
    decision = decide_record_action(
        _record(
            channel="Dr Rangan Chatterjee",
            title="Your Body Keeps Score! - Unhealed Trauma Making You Feel Lost, Addicted, Stressed | Dr. Bessel.txt",
        )
    )

    assert decision.include is True
    assert decision.reason == "strict_channel_authority_signal"


def test_decide_record_action_includes_dhru_purohit_episode_with_expert_signal() -> None:
    decision = decide_record_action(
        _record(
            channel="Dhru Purohit",
            title="If You HEAR THIS, That's A Narcissist Trying To TRAP You! | Dr. Ramani.txt",
        )
    )

    assert decision.include is True
    assert decision.reason == "strict_channel_authority_signal"


def test_decide_record_action_includes_how_to_academy_episode_with_named_clinician() -> None:
    decision = decide_record_action(
        _record(
            channel="How To Academy",
            title="Dr Gabor Mate | Authenticity Can Heal Trauma (Part 2).txt",
        )
    )

    assert decision.include is True
    assert decision.reason == "strict_channel_authority_signal"


def test_decide_record_action_excludes_tedx_talk_without_authority_signal() -> None:
    decision = decide_record_action(
        _record(
            channel="TEDx Talks",
            title="Healing vs. Retaliation: Surviving Trauma and Sexual Abuse | Peter and Adenike Harris.txt",
        )
    )

    assert decision.include is False
    assert decision.reason == "strict_channel_needs_authority_signal"


def test_decide_record_action_excludes_common_ego_without_authority_signal() -> None:
    decision = decide_record_action(
        _record(
            channel="Common Ego",
            title="9 Signs of Covert Narcissism (That Don't Look Narcissistic).txt",
        )
    )

    assert decision.include is False
    assert decision.reason == "strict_channel_needs_authority_signal"


def test_decide_record_action_excludes_blocked_channel() -> None:
    decision = decide_record_action(
        _record(
            channel="Jimmy Kimmel Live",
            title="Trauma Expert Visit.txt",
        )
    )

    assert decision.include is False
    assert decision.reason == "blocked_channel"


def test_decide_record_action_excludes_low_confidence_derivative_channel() -> None:
    decision = decide_record_action(
        _record(
            channel="THE MOTIVATIONAL MIND",
            title="Trauma Expert Explains Narcissistic Abuse.txt",
        )
    )

    assert decision.include is False
    assert decision.reason == "blocked_channel"


def test_curate_youtube_output_writes_curated_dataset_and_report(tmp_path: Path) -> None:
    input_dir = tmp_path / "youtube_output"
    input_dir.mkdir()
    (input_dir / "trusted.jsonl").write_text(
        json.dumps(_record(channel="Heidi Priebe", title="10 Survival Lies You May Tell If You Have CPTSD.txt"))
        + "\n",
        encoding="utf-8",
    )
    (input_dir / "mixed.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    _record(
                        channel="The Diary Of A CEO",
                        title="The Body Trauma Expert: This Eye Movement Trick Can Fix Your Trauma!.txt",
                    )
                ),
                json.dumps(
                    _record(
                        channel="The Diary Of A CEO",
                        title=(
                            "Former FBI Agent: If They Do This Please RUN! "
                            "Narcissists Favourite Trick To Control You!.txt"
                        ),
                    )
                ),
                json.dumps(
                    _record(
                        channel="Mel Robbins",
                        title=(
                            "#1 Neurosurgeon: How to Manifest Anything You Want "
                            "& Unlock the Unlimited Power of Your Mind.txt"
                        ),
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (input_dir / "blocked.jsonl").write_text(
        json.dumps(_record(channel="MSNBC", title="Daily news roundup.txt")) + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "curated.jsonl"
    report_path = tmp_path / "report.json"

    stats = curate_youtube_output(
        input_dir=input_dir,
        output_path=output_path,
        report_path=report_path,
    )

    curated_lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(curated_lines) == EXPECTED_INCLUDED_RECORDS
    assert stats.included_records == EXPECTED_INCLUDED_RECORDS
    assert stats.excluded_records == EXPECTED_EXCLUDED_RECORDS + 1
    assert report["included_records"] == EXPECTED_INCLUDED_RECORDS
    assert report["excluded_records"] == EXPECTED_EXCLUDED_RECORDS + 1
    assert report["decision_counts"]["trusted_channel"] == 1
    assert report["decision_counts"]["strict_channel_authority_signal"] == 1
    assert report["decision_counts"]["strict_channel_needs_authority_signal"] == 1
    assert report["decision_counts"]["strict_channel_low_confidence_title"] == 1
    assert report["decision_counts"]["blocked_channel"] == 1
