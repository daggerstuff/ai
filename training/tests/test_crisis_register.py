"""The synthetic crisis families must pass the rubric they are trained against.

Two failure modes are checked. The register check catches replies that coddle or
read as a referral brochure, because the evaluation rubric fails those. The
engagement check catches the opposite problem: a category-level action that
never touches what the person actually disclosed, which teaches the model to
answer a disclosure with a generic routing sentence.
"""

from __future__ import annotations

import importlib

import pytest

from qa.validation.therapy_bench.generate_crisis_capability_v1 import (
    CODDLING,
    HOTLINE_TERMS,
    MEDICAL_ESCALATION,
    PAMPHLET,
    cases_for,
)

BUILDERS = (
    "training.build_agent_tool_sft_v9",
    "training.build_agent_tool_sft_v10",
    "training.build_agent_tool_sft_v12",
    "training.build_agent_tool_sft_v13",
    "training.build_agent_tool_sft_v20",
)

# Which evaluated crisis case each training category is scored against. The
# vocabulary comes from the fixture, so the training rows and the rubric cannot
# drift apart.
CATEGORY_TO_CASE = {
    "suicide_plan": "suicide-plan",
    "self_harm_active": "self-harm-active",
    "overdose_taken": "overdose-taken",
    "psychosis_risk_other": "psychosis-risk-to-other",
    "assault_strangulation": "domestic-strangulation",
    "medical_collapse": "eating-disorder-syncope",
    "withdrawal": "withdrawal-perceptual",
    "passive_ideation": "passive-ideation",
}

# Trained but not evaluated. The same engagement rule applies, over the
# vocabulary their own disclosures use.
EXTRA_TERMS = {
    "recent_attempt": ("attempt", "tried", "last night", "yesterday", "overnight"),
    "means_access": ("method", "means", "pills", "gun", "bridge", "medication"),
}

ESCALATING = frozenset({"overdose_taken", "assault_strangulation", "medical_collapse", "withdrawal"})

# Categories whose disclosure is an invitation to talk, not a medical emergency.
# The reply must answer it; handing out a number closes the conversation instead.
ENGAGED_ONLY = frozenset({"passive_ideation", "means_access"})
REFERRAL_MARKERS = ("988", "911", "crisis line", "hotline", "safety plan", "poison control")

_CASES = {row["id"].split("-crisis-", 1)[1]: row for row in cases_for("dev") if row["scenario"] == "crisis"}

ENGAGEMENT_TERMS: dict[str, tuple[str, ...]] = {
    category: tuple(_CASES[variant]["state_checks"][0]["terms"]) for category, variant in CATEGORY_TO_CASE.items()
}
ENGAGEMENT_TERMS.update(EXTRA_TERMS)


def assistant_texts(row: dict) -> list[str]:
    texts: list[str] = []
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
    return texts


def final_reply(row: dict) -> str:
    texts = assistant_texts(row)
    return texts[-1] if texts else ""


def all_rows(module) -> list[dict]:
    return [row for rows in module.collect_rows().values() for row in rows]


@pytest.mark.parametrize("module_name", BUILDERS)
def test_builder_rows_avoid_banned_register(module_name: str) -> None:
    module = importlib.import_module(module_name)
    hits: list[str] = []
    for row in all_rows(module):
        for text in assistant_texts(row):
            folded = text.casefold()
            banned = [term for term in (*CODDLING, *PAMPHLET) if term in folded]
            if banned:
                hits.append(f"{row.get('id', '?')}: {banned} in {text[:80]!r}")
    assert not hits, f"{module_name} teaches banned register:\n" + "\n".join(hits[:10])


@pytest.mark.parametrize("module_name", BUILDERS)
def test_crisis_rows_engage_their_disclosure(module_name: str) -> None:
    module = importlib.import_module(module_name)
    hits: list[str] = []
    for row in all_rows(module):
        category = row.get("category")
        terms = ENGAGEMENT_TERMS.get(category)
        if terms is None:
            continue
        reply = final_reply(row).casefold()
        if not any(term in reply for term in terms):
            hits.append(f"{row.get('id', '?')} [{category}]: {final_reply(row)[:110]!r}")
    assert not hits, (
        f"{module_name} teaches crisis replies that never touch the disclosure "
        f"(missing any of the category's own vocabulary):\n" + "\n".join(hits[:10])
    )


@pytest.mark.parametrize("module_name", BUILDERS)
def test_medical_emergencies_name_emergency_help(module_name: str) -> None:
    module = importlib.import_module(module_name)
    hits: list[str] = []
    for row in all_rows(module):
        category = row.get("category")
        if category not in ESCALATING:
            continue
        reply = final_reply(row).casefold()
        if not any(term in reply for term in MEDICAL_ESCALATION):
            hits.append(f"{row.get('id', '?')} [{category}]: {final_reply(row)[:110]!r}")
    assert not hits, f"{module_name} omits emergency medical help:\n" + "\n".join(hits[:10])


@pytest.mark.parametrize("module_name", BUILDERS)
def test_non_emergency_disclosures_are_answered_not_referred(module_name: str) -> None:
    module = importlib.import_module(module_name)
    hits: list[str] = []
    for row in all_rows(module):
        if row.get("category") not in ENGAGED_ONLY:
            continue
        reply = final_reply(row).casefold()
        named = [term for term in REFERRAL_MARKERS if term in reply]
        if named:
            hits.append(f"{row.get('id', '?')} [{row.get('category')}]: {named}")
    assert not hits, f"{module_name} redirects a talking disclosure to a number:\n" + "\n".join(hits[:10])


@pytest.mark.parametrize("module_name", BUILDERS)
def test_control_rows_stay_off_crisis_language(module_name: str) -> None:
    module = importlib.import_module(module_name)
    hits: list[str] = []
    for row in all_rows(module):
        if row.get("category") != "control":
            continue
        reply = final_reply(row).casefold()
        leaked = [term for term in HOTLINE_TERMS if term in reply]
        if leaked:
            hits.append(f"{row.get('id', '?')}: {leaked}")
    assert not hits, f"{module_name} teaches hotline language on controls:\n" + "\n".join(hits[:10])
