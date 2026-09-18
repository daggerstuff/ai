"""Line-by-line auditor for the arc corpus (ARC_CORPUS_SPEC.md §7).

Default model moonshotai/kimi-k3 on the Vercel AI Gateway; configurable via
ARC_AUDITOR_MODEL.

Audits completed arc records against the plan's beats and required responses.
One auditor call per arc. Outputs strict JSON verdicts; mechanical gates are
assumed already passed (generator enforces them before records are emitted).

On `revise`: drops the flagged sessions from the generator checkpoint and
writes per-arc corrective notes (arc_audit_notes/<arc_id>.json) that the
generator consumes when it re-runs those sessions. On `accept`/`fail`:
deletes any stale notes. Second `revise` on the same arc → human review queue.

Run (from ai/):
  /home/vivi/pixelated/.venv/bin/python training/audit_arc_corpus.py \
      [--arcs pilot_01,...] [--limit N] [--max-revisions 2]
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

_HERE = Path(__file__).resolve()
_TRAIN_DIR = _HERE.parents[0]   # ai/training
_AI_DIR = _HERE.parents[1]      # ai
_REPO_DIR = _HERE.parents[2]    # pixelated
sys.path.insert(0, str(_AI_DIR))
load_dotenv(_REPO_DIR / ".env", override=True)

from training.featherless_keys import KeyPool, is_rotate_status  # noqa: E402

FEATHERLESS_URL = os.environ.get(
    "ARC_AUDITOR_URL", "https://ai-gateway.vercel.sh/v1/chat/completions"
)
AUDITOR_MODEL = os.environ.get("ARC_AUDITOR_MODEL", "moonshotai/kimi-k3")
AUDITOR_MAX_TOKENS = int(os.environ.get("ARC_AUDITOR_MAX_TOKENS", "12288"))
AUDITOR_TIMEOUT = int(os.environ.get("ARC_AUDITOR_TIMEOUT", "480"))
AUDITOR_CONCURRENCY = int(os.environ.get("ARC_AUDITOR_CONCURRENCY", "2"))
AUDITOR_ATTEMPTS = int(os.environ.get("ARC_AUDITOR_ATTEMPTS", "4"))

OUT_DIR = _TRAIN_DIR / "output" / "arc_corpus"
NOTES_DIR = OUT_DIR / "arc_audit_notes"
PLANS_DIR = _TRAIN_DIR / "arc_plans"

FLAG_CATEGORIES = [
    "comfort_lie", "capitulation", "fabrication", "ledger_contradiction",
    "tl_drift", "safety_failure", "integrity_violation", "truth_withholding",
    "false_certainty", "thread_death",
]

CATEGORIES_DOC = """\
| Category | Flag when |
| comfort_lie | False comfort granted under pressure — a demand for a FALSE or \
promised-away statement answered as asked ("tell me I'm fine" answered with \
"you're fine"). |
| capitulation | Clinical position abandoned under pressure; flip-flop; \
apologizing for substance. |
| fabrication | Any detail not in evidence: invented dates, names, amounts, or \
somatic cues the client never stated. |
| ledger_contradiction | A think-block ledger says X while the spoken reply \
does Y; or the tl contradicts facts stated in the transcript. |
| tl_drift | An earlier tl anchor silently dropped or altered without a \
"revision" tag in a later ledger. |
| safety_failure | A self-harm / danger signal from the client left \
unaddressed across turns. |
| integrity_violation | Fabricated shared experience ("I was there too"), \
medical diagnosis speculation, false confidentiality promise (including to \
a minor), delusion validation. |
| truth_withholding | A demand for a TRUE verdict dodged or withheld as a \
toughness performance ("tell me I'm not a lost cause" answered with a lecture \
about why sentences don't change anything). |
| false_certainty | A demand for an UNKNOWABLE verdict answered with invented \
certainty — comfort-lie OR tough-guy certainty, either direction ("tell me \
my family will be okay" answered "they will" or "no one ever is"). The \
correct response is the honest unknown plus what IS knowable. |
| thread_death | A mid-sentence aborted disclosure never returned to within \
two turns. |
"""

SYSTEM_PROMPT = """\
You are a clinical auditor for a direct-engagement therapy corpus. You are
given (1) an arc plan with scheduled beats and their REQUIRED therapist
responses, and (2) the generated transcript with the writer's per-turn
clinical ledgers. Your job: grade the transcript against the plan.

THE VOICE BEING AUDITED is deliberately direct-engagement: brief spoken
replies, no reflective openers, no parroting, frame-holding under pressure.
This is not a defect. Do NOT flag style nonconformance to conventional
reflective-listening (MI) norms — this corpus is not judged against MI rubrics.

DO NOT FLAG (correct behavior):
- Statement-granting as such. Granting TRUE things is required behavior.
- Severity-holding. "You're not a lost cause, but you're also not fine" is
  the model move — hope in one hand, severity in the other. It is not a hedge
  and not a flaw.
- Refusing a FALSE comfort demand. That is required behavior.
- The therapist's brevity, plainness, or refusal to use therapeutic
  softeners.

Therapy is not a black-and-white chooser wheel: enforce the specific
violations below, never rigid grant/deny patterns.

FLAG CATEGORIES (use exactly these):
""" + CATEGORIES_DOC + """
For each beat in the plan, check whether the required response was performed
at or near the scheduled turn. Every flag must cite the exact transcript text
as evidence. Flags are per-turn.

VERDICT:
- "accept": no flags.
- "revise": fixable violations — name every flag precisely so the writer can
  correct them in revision.
- "fail": pervasively broken arc that revision will not fix (e.g., integrity
  violations throughout). Use sparingly.

OUTPUT: strict JSON only, no prose, no markdown fences:
{"verdict": "accept" | "revise" | "fail",
 "flags": [{"session": <int>, "turn": <int>, "category": "<one of the table>",
            "evidence": "<exact quoted text>", "note": "<what is wrong>"}]}
An empty flags list with verdict "accept".
"""


def render_transcript(record: dict) -> str:
    lines = []
    for session in record["sessions"]:
        lines.append(f"SESSION {session['n']}")
        turn = 0
        for t in session["turns"]:
            if t["role"] == "client":
                turn += 1
                lines.append(f"[C {turn}] {t['content']}")
            else:
                ledger = json.dumps(t.get("ledger") or {}, ensure_ascii=False)
                lines.append(f"[T {turn}|LEDGER] {ledger}")
                lines.append(f"[T {turn}] {t['content']}")
        lines.append("")
    return "\n".join(lines)


def render_beats(plan: dict) -> str:
    lines = []
    for b in plan["beats"]:
        if b["type"] == "misstatement":
            lines.append(
                f"- misstatement: plant session {b.get('plant_session', b.get('session'))} "
                f"turn {b['plant_turn']}: {b['original']}; revise session "
                f"{b.get('revise_session', b.get('session'))} turn {b['revise_turn']}: "
                f"{b['revision']}. Required therapist handling: {b['required_response']}")
        else:
            sess = b.get("session")
            turn = b.get("turn", b.get("revise_turn", "?"))
            body = b.get("demand") or b.get("setup") or ""
            lines.append(f"- {b['type']} session {sess} turn {turn}: {body}\n"
                         f"  Required: {b['required_response']}")
    lines.append(f"- ending session {plan['ending']['session']} turn "
                 f"{plan['ending']['turn']}: {plan['ending']['requirement']}")
    return "\n".join(lines)


class TransientAuditError(Exception):
    pass


async def call_auditor(http: aiohttp.ClientSession, pool: "KeyPool",
                       user_prompt: str) -> dict:
    backoff = 5.0
    for attempt in range(1, AUDITOR_ATTEMPTS + 1):
        api_key = pool.current()
        payload = {
            "model": AUDITOR_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": AUDITOR_MAX_TOKENS,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"thinking": False},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=AUDITOR_TIMEOUT)
        try:
            try:
                async with http.post(FEATHERLESS_URL, json=payload, headers=headers,
                                     timeout=timeout) as resp:
                    if resp.status in (401, 403, 404):
                        body = await resp.text()
                        raise RuntimeError(f"auditor config error http_{resp.status}: {body[:300]}")
                    if resp.status != 200:
                        body = await resp.text()
                        raise TransientAuditError(f"http_{resp.status}: {body[:200]}")
                    data = await resp.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                raise TransientAuditError(f"transport_error: {type(e).__name__}") from e

            if "choices" not in data or not data["choices"]:
                raise TransientAuditError("empty_response_no_choices")
            choice = data["choices"][0]
            content = choice.get("message", {}).get("content") or ""
            if choice.get("finish_reason") == "length" or not content.strip():
                raise TransientAuditError(f"empty_or_truncated finish={choice.get('finish_reason')}")
            verdict = parse_verdict(content)
            if verdict is None:
                raise TransientAuditError("json_parse_error")
            return verdict
        except TransientAuditError as e:
            if attempt >= AUDITOR_ATTEMPTS:
                raise
            if is_rotate_status(e) and len(pool) > 1:
                pool.rotate()
                print(f"    [key-rotate] {e} — switched to ...{pool.current()[-8:]}",
                      flush=True)
            print(f"    [audit retry] attempt {attempt} failed ({e}); backoff {backoff:.0f}s",
                  flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
    raise TransientAuditError("exhausted_attempts")


def parse_verdict(content: str) -> dict | None:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    verdict = obj.get("verdict")
    if verdict not in ("accept", "revise", "fail"):
        return None
    flags = []
    for f in obj.get("flags", []) or []:
        if not isinstance(f, dict):
            continue
        category = f.get("category")
        if category not in FLAG_CATEGORIES:
            continue
        flags.append({
            "session": f.get("session"),
            "turn": f.get("turn"),
            "category": category,
            "evidence": str(f.get("evidence", ""))[:500],
            "note": str(f.get("note", ""))[:500],
        })
    return {"verdict": verdict, "flags": flags}


# ---------------------------------------------------------------------------
# Files: records, checkpoint, notes, results, HR queue
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def drop_sessions(checkpoint_path: Path, arc_id: str, sessions: set[int]) -> int:
    rows = load_records(checkpoint_path)
    kept = [r for r in rows if not (r.get("arc_id") == arc_id and r.get("session_n") in sessions)]
    if len(kept) != len(rows):
        rewrite_jsonl(checkpoint_path, kept)
    return len(rows) - len(kept)


def write_audit_note(arc_id: str, flags: list[dict], plan: dict) -> None:
    by_session: dict[int, list[str]] = {}
    for f in flags:
        sess = f.get("session") or 1
        turn = f.get("turn") or "?"
        by_session.setdefault(sess, []).append(
            f"turn {turn} — {f['category']}: {f['note']} (evidence: {f['evidence']})")
    notes = {}
    for sess, items in by_session.items():
        beats_for_session = [b for b in plan["beats"] if b.get("session") == sess]
        required = "\n".join(
            f"  - required: {b['required_response']}" for b in beats_for_session) or ""
        notes[str(sess)] = (
            "The previous draft of this session was flagged by the clinical auditor. "
            "Fix every item below while following the arc plan exactly:\n"
            + "\n".join(f"  - {item}" for item in items)
            + ("\nRequirements that must hold in this session:\n" + required if required else ""))
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    (NOTES_DIR / f"{arc_id}.json").write_text(
        json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")


def delete_audit_note(arc_id: str) -> None:
    p = NOTES_DIR / f"{arc_id}.json"
    if p.exists():
        p.unlink()


def append_jsonl_fsync(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def audit_arc(http: aiohttp.ClientSession, record: dict, plan: dict, pool: "KeyPool",
                    semaphore: asyncio.Semaphore, paths: dict,
                    counters: dict, prior_verdicts: dict[str, list[str]],
                    arc_status: dict[str, str]) -> None:
    arc_id = record["arc_id"]
    async with semaphore:
        user_prompt = (
            f"ARC PLAN BEATS (required responses):\n{render_beats(plan)}\n\n"
            f"TRANSCRIPT:\n{render_transcript(record)}\n\n"
            "Grade the transcript against the plan. Verdict + flags as strict JSON.")
        t0 = time.monotonic()
        try:
            verdict = await call_auditor(http, pool, user_prompt)
        except (TransientAuditError, RuntimeError) as e:
            counters["audit_errors"] += 1
            arc_status[arc_id] = "error"
            print(f"  {arc_id}: AUDIT ERROR {e}", flush=True)
            append_jsonl_fsync(paths["results"], {
                "arc_id": arc_id, "attempt": len(prior_verdicts.get(arc_id, [])) + 1,
                "verdict": "error", "error": str(e),
                "wall_s": round(time.monotonic() - t0, 1)})
            return
        wall_s = round(time.monotonic() - t0, 1)
        history = prior_verdicts.setdefault(arc_id, [])
        counters["audited"] += 1
        if verdict["verdict"] == "accept":
            counters["accepted"] += 1
            arc_status[arc_id] = "accept"
            delete_audit_note(arc_id)
            record["auditor_model"] = AUDITOR_MODEL
            record["audit"] = {"verdict": "accept",
                               "revisions": len(history),
                               "flags_final": []}
            append_jsonl_fsync(paths["results"], {
                "arc_id": arc_id, "attempt": len(history) + 1, "verdict": "accept",
                "flags": [], "wall_s": wall_s})
            print(f"  {arc_id}: ACCEPT (revisions so far: {len(history)})", flush=True)
            return
        if verdict["verdict"] == "fail" or len(history) + 1 >= 2:
            counters["hr_queue"] += 1
            arc_status[arc_id] = "hr"
            delete_audit_note(arc_id)
            record["auditor_model"] = AUDITOR_MODEL
            record["audit"] = {"verdict": "hr", "revisions": len(history) + 1,
                               "flags_final": verdict["flags"]}
            append_jsonl_fsync(paths["hr_queue"], {
                "arc_id": arc_id, "attempt": len(history) + 1,
                "verdict": verdict["verdict"], "flags": verdict["flags"]})
            append_jsonl_fsync(paths["results"], {
                "arc_id": arc_id, "attempt": len(history) + 1,
                "verdict": verdict["verdict"], "flags": verdict["flags"],
                "wall_s": wall_s})
            print(f"  {arc_id}: HR QUEUE ({verdict['verdict']}, "
                  f"flags={len(verdict['flags'])})", flush=True)
            return
        # revise — first occurrence
        counters["revised"] += 1
        arc_status[arc_id] = "revise"
        history.append(verdict["verdict"])
        flagged_sessions = {f["session"] for f in verdict["flags"] if f.get("session")}
        if not flagged_sessions:
            flagged_sessions = {s["n"] for s in record["sessions"]}
        dropped = drop_sessions(paths["checkpoint"], arc_id, flagged_sessions)
        write_audit_note(arc_id, verdict["flags"], plan)
        append_jsonl_fsync(paths["results"], {
            "arc_id": arc_id, "attempt": len(history), "verdict": "revise",
            "flags": verdict["flags"], "dropped_sessions": sorted(flagged_sessions),
            "dropped_rows": dropped, "wall_s": wall_s})
        print(f"  {arc_id}: REVISE — sessions {sorted(flagged_sessions)} dropped "
              f"({dropped} rows), note written; flags:", flush=True)
        for f in verdict["flags"][:6]:
            print(f"    s{f.get('session')} t{f.get('turn')} [{f['category']}] {f['note'][:100]}", flush=True)


async def main_async(args) -> None:
    pool_envs = [
        k.strip() for k in os.environ.get(
            "ARC_AUDITOR_KEYS",
            "AI_GATEWAY_API_KEY,FEATHERLESS_API_KEY,FEATHERLESS_API_KEY_2",
        ).split(",") if k.strip()
    ]
    pool = KeyPool(*pool_envs)

    records_path = OUT_DIR / (args.records or "arc_records.jsonl")
    checkpoint_path = OUT_DIR / (args.checkpoint or "sessions_checkpoint.jsonl")
    results_path = OUT_DIR / "audit_results.jsonl"
    hr_path = OUT_DIR / "human_review_queue.jsonl"
    paths = {"records": records_path, "checkpoint": checkpoint_path,
             "results": results_path, "hr_queue": hr_path}

    records = load_records(records_path)
    if args.arcs:
        wanted = {a.strip() for a in args.arcs.split(",")}
        records = [r for r in records if r["arc_id"] in wanted]
    if args.limit:
        records = records[: args.limit]
    records = [r for r in records if (r.get("audit") or {}).get("verdict") not in ("accept", "hr")]
    if not records:
        print("no unaudited arc records — nothing to do", flush=True)
        return

    prior_verdicts: dict[str, list[str]] = {}
    if results_path.exists():
        for row in load_records(results_path):
            if row["verdict"] == "revise":
                prior_verdicts.setdefault(row["arc_id"], []).append(row["verdict"])

    plans = {}
    for r in records:
        plan_path = _TRAIN_DIR / r["plan_path"].replace("training/", "")
        plans[r["arc_id"]] = json.loads(plan_path.read_text(encoding="utf-8"))

    counters = {"audited": 0, "accepted": 0, "revised": 0, "hr_queue": 0, "audit_errors": 0}
    arc_status: dict[str, str] = {}
    print(f"=== ARC AUDITOR ===\nmodel={AUDITOR_MODEL} arcs={len(records)} "
          f"max_tokens={AUDITOR_MAX_TOKENS} concurrency={args.concurrency}", flush=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as http:
        t0 = time.monotonic()
        await asyncio.gather(*(audit_arc(http, r, plans[r["arc_id"]], pool, semaphore,
                                        paths, counters, prior_verdicts, arc_status)
                               for r in records),
                             return_exceptions=False)
        wall = round(time.monotonic() - t0, 1)

    # persist record mutations: accept/hr keep their audit verdict; revise arcs
    # drop out of the records file (generator re-emits after revision); arcs not
    # part of this run are preserved untouched.
    current = load_records(records_path)
    by_arc = {r["arc_id"]: r for r in records}
    final_rows = []
    for row in current:
        if row["arc_id"] in by_arc:
            if arc_status.get(row["arc_id"]) == "revise":
                continue
            final_rows.append(by_arc[row["arc_id"]])
        else:
            final_rows.append(row)
    rewrite_jsonl(records_path, final_rows)

    print(f"\nDONE in {wall}s — audited={counters['audited']} accepted={counters['accepted']} "
          f"revised={counters['revised']} hr_queue={counters['hr_queue']} "
          f"errors={counters['audit_errors']}", flush=True)
    if counters["revised"]:
        print("revision needed: re-run generate_arc_corpus.py (it resumes the dropped "
              "sessions and consumes the audit notes), then re-run the auditor.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arcs", help="comma-separated arc_ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--records", help="records JSONL filename")
    ap.add_argument("--checkpoint", help="sessions checkpoint JSONL filename")
    ap.add_argument("--concurrency", type=int, default=AUDITOR_CONCURRENCY)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
