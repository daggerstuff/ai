"""The synthetic crisis families must not teach the register the rubric bans.

The evaluation rubric fails a reply for coddling or for reading as a referral
brochure. These builders generate the training rows for those same scenarios, so
a reply template that opens with "that is a lot to carry" or names a country
menu teaches the model to fail the check it is supposed to pass.
"""

from __future__ import annotations

import importlib

import pytest

from qa.validation.therapy_bench.generate_crisis_capability_v1 import CODDLING, PAMPHLET

BUILDERS = (
    "training.build_agent_tool_sft_v9",
    "training.build_agent_tool_sft_v10",
    "training.build_agent_tool_sft_v12",
    "training.build_agent_tool_sft_v13",
)


def assistant_texts(row: dict) -> list[str]:
    texts: list[str] = []
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
    return texts


@pytest.mark.parametrize("module_name", BUILDERS)
def test_builder_rows_avoid_banned_register(module_name: str) -> None:
    module = importlib.import_module(module_name)
    hits: list[str] = []
    for rows in module.collect_rows().values():
        for row in rows:
            for text in assistant_texts(row):
                folded = text.casefold()
                banned = [term for term in (*CODDLING, *PAMPHLET) if term in folded]
                if banned:
                    hits.append(f"{row.get('id', '?')}: {banned} in {text[:80]!r}")
    assert not hits, f"{module_name} teaches banned register:\n" + "\n".join(hits[:10])
