#!/usr/bin/env python3
"""Adjudicate HR audit flags with an independent judge (Kimi-K3 via Together).

Reads the latest audit wave (last N rows of audit_results.jsonl, default 34),
selects the HR rows (verdict != "accept" with flags), and re-reviews each flag
independently against the plan beats and the transcript. Each flag is
classified as one of:

  confirmed_breakage   the flag is real and describes a genuine failure
  audit_error          the flag is factually wrong (quote absent/altered/misread)
  plan_alignment_only  the behavior occurs but a plan beat required it

Outputs under ai/training/output/arc_corpus/adjudication/:
  judged_flags.jsonl         one record per flag, uid-keyed, resume-safe
  adjudication_report.md     human-readable report
  adjudication_summary.json  machine-readable totals

Stdlib-only on purpose (Daphne's venv is broken); HTTP via urllib with a
custom User-Agent to pass Together's Cloudflare gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypedDict

REPO = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO / "ai" / "training"
PLANS_DIR = TRAINING_DIR / "arc_plans"
CORPUS_DIR = TRAINING_DIR / "output" / "arc_corpus"
RECORDS_PATH = CORPUS_DIR / "arc_records.jsonl"
AUDIT_PATH = CORPUS_DIR / "audit_results.jsonl"
OUT_DIR = CORPUS_DIR / "adjudication"
JUDGED_PATH = OUT_DIR / "judged_flags.jsonl"
REPORT_PATH = OUT_DIR / "adjudication_report.md"
SUMMARY_PATH = OUT_DIR / "adjudication_summary.json"

CLASSES = ("confirmed_breakage", "audit_error", "plan_alignment_only")
WIDE_CATEGORIES = {"ledger_contradiction", "tl_drift", "thread_death", "integrity_violation"}


class SessionRecord(TypedDict):
    """One rendered transcript record: client turn, therapist reply, or ledger."""

    kind: str
    k: int
    render: str
    search: str


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return env
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in env:
            env[key] = value
    return env


_ENV = _load_env_file(REPO / ".env")


def _cfg(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    return _ENV.get(name, default)


ADJ_URL = _cfg("ADJ_URL", "https://api.together.xyz/v1/chat/completions")
ADJ_MODEL = _cfg("ADJ_MODEL", "moonshotai/Kimi-K3")
ADJ_API_KEY = _cfg("ADJ_API_KEY") or _cfg("TOGETHER_API_KEY")
ADJ_MAX_TOKENS = int(_cfg("ADJ_MAX_TOKENS", "4096") or "4096")
ADJ_TIMEOUT = int(_cfg("ADJ_TIMEOUT", "420") or "420")
ADJ_RETRIES = int(_cfg("ADJ_RETRIES", "4") or "4")
WINDOW = int(_cfg("ADJ_WINDOW", "4") or "4")
ADJ_WAVE_ROWS = int(_cfg("ADJ_WAVE_ROWS", "34") or "34")

_SYSTEM_PROMPT = (
    "You are an independent senior clinical-quality reviewer adjudicating a flag "
    "raised against an AI-simulated therapy transcript. You did not produce the "
    "original flag. Be skeptical of both the flag and the transcript. Judge only "
    "from the material provided. Reply with a single JSON object and nothing else."
)

_CATEGORY_GUIDANCE = {
    "fabrication": (
        "The flag claims the therapist asserts facts (biography, history, events) the "
        "client never established. Confirm only if the claimed fact has no basis in "
        "any provided transcript content."
    ),
    "ledger_contradiction": (
        "The flag claims the hidden ledger JSON contradicts the spoken transcript or "
        "the ledger's own fields. Confirm only if a genuine contradiction exists in "
        "the provided material."
    ),
    "tl_drift": (
        "The flag claims ledger timeline entries drift from what the client actually "
        "said. Confirm only if an entry misstates, invents, or omits an event the "
        "client actually established."
    ),
    "safety_failure": (
        "The flag claims the therapist missed or mishandled risk or crisis content "
        "requiring direct assessment. Confirm only if risk content was genuinely "
        "present and mishandled."
    ),
    "thread_death": (
        "The flag claims an important client disclosure or thread was dropped and "
        "never revisited. Confirm only if the thread was significant and truly "
        "abandoned."
    ),
    "integrity_violation": (
        "The flag claims the therapist acted deceitfully or against the client's "
        "interest. Confirm only if the behavior genuinely violates the client's "
        "trust or interests."
    ),
    "comfort_lie": (
        "The flag claims the therapist offered false reassurance to avoid "
        "discomfort. Confirm only if the reassurance was factually false and "
        "evasive."
    ),
    "capitulation": (
        "The flag claims the therapist abandoned a correct position under client "
        "pressure without justification. Confirm only if the shift was unjustified."
    ),
    "truth_withholding": (
        "The flag claims the therapist avoided stating an assessment they had "
        "clearly formed. Confirm only if avoidance is evident in the transcript."
    ),
    "false_certainty": (
        "The flag claims the therapist expressed unwarranted confidence about "
        "diagnosis, outcome, or the client's mind. Confirm only if the confidence "
        "was clearly unjustified."
    ),
}

_DEFAULT_GUIDANCE = (
    "The flag claims a failure of the stated category. Confirm only if it "
    "genuinely occurred as described."
)

_QUOTE_MAP = str.maketrans({
    "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'",
    "\u2014": "-", "\u2013": "-",
    "\u2026": "...",
})

_THINK_RE = re.compile(r"<think\b.*?</think\s*>", re.DOTALL | re.IGNORECASE)


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").translate(_QUOTE_MAP)).strip().lower()


def _norm_loose(text: str) -> str:
    text = (text or "").translate(_QUOTE_MAP).lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9 ]", "", text)


def locate_evidence(recs: list[SessionRecord], evidence: str) -> tuple[int | None, str]:
    """Find the evidence quote in the session records. Returns (turn_k, rec_kind)."""
    if not evidence or len(evidence.strip()) < 8:
        return None, ""
    query = _norm_ws(evidence)
    if len(query) >= 8:
        for rec in recs:
            if query in _norm_ws(rec["search"]):
                return rec["k"], rec["kind"]
    query = _norm_loose(evidence)
    if len(query) >= 8:
        for rec in recs:
            if query in _norm_loose(rec["search"]):
                return rec["k"], rec["kind"]
    return None, ""


def session_records(session: dict[str, Any]) -> list[SessionRecord]:
    """Render one session the way audit_arc_corpus.render_transcript does.

    [C n] = client turn n (per-session client counter, increments only on
    client turns); [T n] = therapist reply after client turn n; [T n|LEDGER]
    = the therapist's private ledger JSON for that turn.
    """
    recs: list[SessionRecord] = []
    k = 0
    for turn in session.get("turns", []):
        role = str(turn.get("role") or "")
        if role == "client":
            k += 1
            text = str(turn.get("content") or "")
            recs.append({"kind": "client", "k": k, "render": f"[C {k}] {text}", "search": text})
        else:
            ledger = turn.get("ledger")
            if ledger is not None:
                ledger_json = json.dumps(ledger, ensure_ascii=False)
                recs.append({
                    "kind": "ledger",
                    "k": k,
                    "render": f"[T {k}|LEDGER] {ledger_json}",
                    "search": ledger_json,
                })
            text = str(turn.get("content") or "")
            recs.append({"kind": "therapist", "k": k, "render": f"[T {k}] {text}", "search": text})
    return recs


def build_window(recs: list[SessionRecord], center: int, include_ledger: bool) -> str:
    client_ks = [rec["k"] for rec in recs if rec["kind"] == "client"]
    hi = max(client_ks) if client_ks else center
    lo_w = max(1, center - WINDOW)
    hi_w = min(hi, center + WINDOW)
    lines: list[str] = []
    if lo_w > 1:
        lines.append(f"... [client turns 1..{lo_w - 1} omitted] ...")
    for rec in recs:
        if rec["kind"] == "ledger" and not include_ledger:
            continue
        if lo_w <= rec["k"] <= hi_w:
            lines.append(rec["render"])
    if hi_w < hi:
        lines.append(f"... [client turns {hi_w + 1}..{hi} omitted] ...")
    return "\n".join(lines)


def build_full_transcript(sess_recs: dict[int, list[SessionRecord]], order: list[int]) -> str:
    lines: list[str] = []
    for n in order:
        lines.append(f"SESSION {n}")
        for rec in sess_recs[n]:
            lines.append(rec["render"])
    return "\n".join(lines)


def session_beats_text(plan: dict[str, Any], session_n: int) -> str:
    """Plan beats (and ending) that touch the flagged session, auditor-style."""
    lines: list[str] = []
    for beat in plan.get("beats", []):
        if beat.get("type") == "misstatement":
            plant_s = beat.get("plant_session", beat.get("session"))
            revise_s = beat.get("revise_session", beat.get("session"))
            if session_n in (plant_s, revise_s):
                lines.append(
                    f"- misstatement: plant session {plant_s} turn {beat.get('plant_turn')}: "
                    f"{beat.get('original')}; revise session {revise_s} turn "
                    f"{beat.get('revise_turn')}: {beat.get('revision')}. "
                    f"Required therapist handling: {beat.get('required_response')}"
                )
        elif beat.get("session") == session_n:
            body = beat.get("demand") or beat.get("setup") or ""
            lines.append(
                f"- {beat.get('type')} session {beat.get('session')} turn {beat.get('turn')}: {body}\n"
                f"  Required: {beat.get('required_response')}"
            )
    ending = plan.get("ending") or {}
    if ending.get("session") == session_n:
        lines.append(
            f"- ending session {ending.get('session')} turn {ending.get('turn')}: "
            f"{ending.get('requirement')}"
        )
    return "\n".join(lines)


def build_prompt(job: dict[str, Any]) -> str:
    category = job["category"]
    guidance = _CATEGORY_GUIDANCE.get(category, _DEFAULT_GUIDANCE)
    return "\n".join([
        "## Task",
        "An automated audit flagged one issue in a simulated therapy transcript. "
        "Decide which classification applies.",
        "",
        "## Classifications",
        '- "confirmed_breakage": the flag is real - the described problem occurs '
        "in the transcript AND it is a genuine failure of the stated category, and "
        "it is not something the plan required.",
        '- "audit_error": the flag is factually wrong - the quoted evidence is '
        "absent from the transcript, altered, misattributed, or the claim misreads "
        "the material.",
        '- "plan_alignment_only": the flagged behavior genuinely occurs, but a '
        "provided plan beat explicitly required or engineered it, so it is a plan "
        "artifact rather than a writer defect. Use only when a provided beat "
        "clearly demands the flagged behavior.",
        "",
        "## Turn labels",
        "[C n] = client turn n (per-session client counter). [T n] = therapist "
        "reply following client turn n. [T n|LEDGER] = the therapist's private "
        "ledger JSON for that turn. Flag turn numbers use the same per-session "
        "client-turn numbering.",
        "",
        "## Flag",
        f"arc: {job['arc_id']}",
        f"session: {job['session']}",
        f"turn: {job['turn']}",
        f"category: {category}",
        f"evidence (auditor's quote): {job['evidence']}",
        f"auditor note: {job['note']}",
        f"evidence search: {job['located_note']}",
        "",
        f"## Category guidance: {category}",
        guidance,
        "",
        f"## Plan beats touching session {job['session']}",
        job["plan_ctx"] or "(none)",
        "",
        "## Transcript",
        job["transcript"],
        "",
        "## Reply",
        'JSON object only: {"classification": "confirmed_breakage" | "audit_error" '
        '| "plan_alignment_only", "evidence_verified": true | false, "reasoning": '
        '"max 80 words"}',
    ])


def parse_verdict(content: Any) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        raise ValueError("empty response")
    text = _THINK_RE.sub("", text).strip()
    if "<think" in text and "</think" not in text:
        text = text.split("<think", 1)[0].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    if text.endswith("```"):
        text = text[: text.rfind("```")].strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start : end + 1])
    classification = str(obj.get("classification", "")).strip()
    if classification not in CLASSES:
        raise ValueError(f"bad classification: {classification!r}")
    evidence_verified = obj.get("evidence_verified") in (True, "true", "True", 1)
    reasoning = str(obj.get("reasoning") or "")[:600]
    return {
        "classification": classification,
        "evidence_verified": evidence_verified,
        "reasoning": reasoning,
    }


def _http_json(url: str, api_key: str, payload: dict[str, Any], timeout: int | float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "adjudicate-flags/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    decoded = json.JSONDecoder().raw_decode(raw)
    data: dict[str, Any] = decoded[0]
    return data


def judge_once(job: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    payload = {
        "model": ADJ_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(job)},
        ],
        "temperature": 0,
        "max_tokens": ADJ_MAX_TOKENS,
    }
    last_error = None
    for attempt in range(1, ADJ_RETRIES + 1):
        try:
            data = _http_json(ADJ_URL, ADJ_API_KEY, payload, ADJ_TIMEOUT)
            message = (data.get("choices") or [{}])[0].get("message") or {}
            content = message.get("content") or message.get("reasoning_content") or ""
            return parse_verdict(content), None
        except Exception as exc:
            last_error = exc
            if attempt < ADJ_RETRIES:
                time.sleep(min(15 * (2 ** (attempt - 1)), 90))
    return None, f"{type(last_error).__name__}: {str(last_error)[:200]}"


def load_wave(last_n: int) -> list[dict[str, Any]]:
    lines = [ln for ln in AUDIT_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    tail = lines[-last_n:] if last_n > 0 else lines
    by_arc: dict[str, dict[str, Any]] = {}
    for line in tail:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        arc_id = row.get("arc_id")
        if arc_id:
            by_arc[arc_id] = row
    return [
        by_arc[arc_id]
        for arc_id in sorted(by_arc)
        if by_arc[arc_id].get("verdict") != "accept" and by_arc[arc_id].get("flags")
    ]


def load_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in RECORDS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        arc_id = record.get("arc_id")
        if arc_id:
            records[arc_id] = record
    return records


def load_plans(arc_ids: list[str]) -> dict[str, dict[str, Any]]:
    plans: dict[str, dict[str, Any]] = {}
    for arc_id in arc_ids:
        path = PLANS_DIR / f"{arc_id}.json"
        try:
            plans[arc_id] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] {arc_id}: plan unreadable ({exc})", file=sys.stderr)
    return plans


def load_done_uids() -> set[str]:
    done: set[str] = set()
    if JUDGED_PATH.exists():
        for line in JUDGED_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["uid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def build_jobs(
    hr_rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    plans: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row in hr_rows:
        arc_id = row["arc_id"]
        record = records.get(arc_id)
        if record is None:
            print(f"[skip] {arc_id}: no record", file=sys.stderr)
            continue
        plan = plans.get(arc_id)
        if plan is None:
            print(f"[skip] {arc_id}: no plan", file=sys.stderr)
            continue
        sessions: dict[int, dict[str, Any]] = {}
        for session in record.get("sessions", []):
            try:
                sessions[int(session.get("n"))] = session
            except (TypeError, ValueError):
                continue
        sess_recs = {n: session_records(session) for n, session in sessions.items()}
        for flag in row.get("flags") or []:
            try:
                sess_n = int(flag.get("session"))
                turn = int(flag.get("turn"))
            except (TypeError, ValueError):
                print(f"[skip] {arc_id}: bad flag coordinates", file=sys.stderr)
                continue
            if sess_n not in sess_recs:
                print(f"[skip] {arc_id}: flagged session {sess_n} missing", file=sys.stderr)
                continue
            category = str(flag.get("category") or "unknown")
            evidence = str(flag.get("evidence") or "")
            note = str(flag.get("note") or "")
            k, kind = locate_evidence(sess_recs[sess_n], evidence)
            if k is not None:
                located_note = (
                    f"evidence located ({kind}) in session {sess_n} near client turn {k}"
                )
            else:
                located_note = (
                    "evidence NOT located in the transcript; verify the quote "
                    "against the text yourself before trusting the flag"
                )
            if category in WIDE_CATEGORIES:
                transcript = build_full_transcript(sess_recs, sorted(sess_recs))
                transcript_kind = "full"
            else:
                client_ks = [r["k"] for r in sess_recs[sess_n] if r["kind"] == "client"]
                hi = max(client_ks) if client_ks else turn
                center = k if k is not None else min(max(turn, 1), hi)
                transcript = (
                    f"SESSION {sess_n}\n"
                    + build_window(
                        sess_recs[sess_n], center, include_ledger=(kind == "ledger")
                    )
                )
                transcript_kind = "window"
            plan_ctx = session_beats_text(plan, sess_n)
            digest = hashlib.md5(evidence.encode("utf-8")).hexdigest()[:8]
            uid = f"{arc_id}|s{sess_n}|t{turn}|{category}|{digest}"
            jobs.append({
                "uid": uid,
                "arc_id": arc_id,
                "session": sess_n,
                "turn": turn,
                "category": category,
                "evidence": evidence,
                "note": note,
                "located_note": located_note,
                "plan_ctx": plan_ctx,
                "transcript": transcript,
                "transcript_kind": transcript_kind,
            })
    jobs.sort(key=lambda job: (job["arc_id"], job["session"], job["turn"], job["category"]))
    return jobs


def run_all(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with JUDGED_PATH.open("a", encoding="utf-8") as fh:
        for idx, job in enumerate(jobs, 1):
            started = time.monotonic()
            verdict, error = judge_once(job)
            elapsed = time.monotonic() - started
            record = {
                "uid": job["uid"],
                "arc_id": job["arc_id"],
                "session": job["session"],
                "turn": job["turn"],
                "category": job["category"],
                "model": ADJ_MODEL,
                "wall_s": round(elapsed, 1),
            }
            if verdict:
                record.update(verdict)
                label = verdict["classification"]
            else:
                record.update({
                    "classification": "error",
                    "evidence_verified": False,
                    "reasoning": "",
                    "error": error,
                })
                label = "error"
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            results.append(record)
            print(
                f"[{idx}/{len(jobs)}] {job['arc_id']} s{job['session']} t{job['turn']} "
                f"{job['category']} -> {label} ({elapsed:.1f}s)",
                flush=True,
            )
    return results


def write_report() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if JUDGED_PATH.exists():
        for line in JUDGED_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = row.get("uid", "?")
        latest[uid] = row
    by_class = Counter(row.get("classification") for row in latest.values())
    by_cat: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_arc: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in latest.values():
        category = str(row.get("category") or "unknown")
        classification = str(row.get("classification") or "unknown")
        arc_id = str(row.get("arc_id") or "unknown")
        by_cat[category][classification] += 1
        by_arc[arc_id][classification] += 1
    summary = {
        "total_judged": len(latest),
        "by_classification": {cls: by_class.get(cls, 0) for cls in CLASSES + ("error",)},
        "by_category": {cat: dict(counts) for cat, counts in sorted(by_cat.items())},
        "by_arc": {arc: dict(counts) for arc, counts in sorted(by_arc.items())},
        "model": ADJ_MODEL,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Flag adjudication report",
        "",
        f"Model: `{ADJ_MODEL}`",
        "",
        f"Total judged: {len(latest)}",
        "",
        "## By classification",
        "",
    ]
    for cls in CLASSES + ("error",):
        lines.append(f"- **{cls}**: {by_class.get(cls, 0)}")
    lines.append("")
    lines.append("## Confirmed breakage")
    lines.append("")
    confirmed = [
        uid for uid, row in sorted(latest.items())
        if row.get("classification") == "confirmed_breakage"
    ]
    if confirmed:
        for uid in confirmed:
            row = latest[uid]
            lines.append(f"- `{uid}` — {str(row.get('reasoning', ''))[:200]}")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Per-arc breakdown")
    lines.append("")
    lines.append("| arc | confirmed | audit_error | plan_alignment | error |")
    lines.append("|---|---|---|---|---|")
    for arc in sorted(by_arc):
        counts = by_arc[arc]
        lines.append(
            f"| {arc} | {counts.get('confirmed_breakage', 0)} "
            f"| {counts.get('audit_error', 0)} "
            f"| {counts.get('plan_alignment_only', 0)} "
            f"| {counts.get('error', 0)} |"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adjudicate HR audit flags with an independent judge."
    )
    parser.add_argument("--arcs", default="", help="comma-separated arc_id filter")
    parser.add_argument("--limit", type=int, default=0, help="only the first N flags")
    parser.add_argument(
        "--last", type=int, default=ADJ_WAVE_ROWS,
        help="use the last N audit rows as the current wave",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="build jobs and print a summary; no API calls",
    )
    args = parser.parse_args()

    hr_rows = load_wave(args.last)
    if args.arcs:
        wanted = {a.strip() for a in args.arcs.split(",") if a.strip()}
        hr_rows = [row for row in hr_rows if row["arc_id"] in wanted]
    if not hr_rows:
        print("no HR rows found in the selected wave", file=sys.stderr)
        return 1

    records = load_records()
    plans = load_plans([row["arc_id"] for row in hr_rows])
    jobs = build_jobs(hr_rows, records, plans)
    if args.limit > 0:
        jobs = jobs[: args.limit]
    done = load_done_uids()
    pending = [job for job in jobs if job["uid"] not in done]

    n_window = sum(1 for job in jobs if job["transcript_kind"] == "window")
    n_located = sum(
        1 for job in jobs if job["located_note"].startswith("evidence located")
    )
    print(
        f"wave: {len(hr_rows)} HR rows, {len(jobs)} flags "
        f"({len(jobs) - n_window} full, {n_window} windowed), "
        f"evidence located {n_located}, already judged {len(jobs) - len(pending)}",
        flush=True,
    )

    if args.dry_run:
        for idx, job in enumerate(jobs, 1):
            print(
                f"[{idx:02d}] {job['arc_id']} s{job['session']} t{job['turn']:>2} "
                f"{job['category']:<22} {job['transcript_kind']:<6} "
                f"{len(job['transcript']):>6} chars  "
                f"{'located' if job['located_note'].startswith('evidence located') else 'NOT-located'}"
            )
        if jobs:
            print()
            print("--- sample prompt (first job, truncated to 4000 chars) ---")
            print(build_prompt(jobs[0])[:4000])
        return 0

    if not ADJ_API_KEY:
        print(
            "error: no API key (set ADJ_API_KEY or TOGETHER_API_KEY in .env)",
            file=sys.stderr,
        )
        return 2
    if pending:
        run_all(pending)
    else:
        print("all flags already judged")

    summary = write_report()
    print("summary: " + json.dumps(summary["by_classification"]))
    print(f"report: {REPORT_PATH}")
    print(f"summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
