"""Arc-plan lint: schema, provenance, and anti-pattern checks for arc plans.

Self-contained (no dependency on build_pilot_arc_plans). Checks the contract
the writer (generate_arc_corpus.py) and auditor (audit_arc_corpus.py) rely on:

  - schema: required keys, beat types, per-type beat keys, session budgets
  - provenance: timeline anchor format, told/claim/untold markers, ordering
  - anti-patterns: the "untold diagnostic" guard (the pilot_06 MCI bug class),
    crisis language with no safety beat, beat collisions

Exports lint_plan(plan) -> (errors, warnings) and
lint_batch(plans, paths) -> (errors, warnings) for reuse by build_arc_plans.py
(the LLM plan generator lints every candidate before writing it to disk).

Run (from ai/):
  /home/vivi/pixelated/.venv/bin/python training/lint_arc_plans.py [paths...]
      [--strict] [--no-set-level]
No paths = training/arc_plans/*.json.
Exit: 0 clean, 1 errors (or warnings under --strict), 2 usage error.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, TypeGuard

_HERE = Path(__file__).resolve()
_TRAIN_DIR = _HERE.parents[0]   # ai/training
PLANS_DIR = _TRAIN_DIR / "arc_plans"

# Rule thresholds.
AGE_MAX = 120
SESSION_TURNS_MIN = 8
SESSION_TURNS_MAX = 40
TOTAL_TURNS_MIN = 16
TOTAL_TURNS_MAX = 120
MIN_TIMELINE_EVENTS = 2
MIN_SESSIONS = 2
MIN_SURFACE_TURN = 3
MIN_PRESSURE_TYPES = 2
SET_LEVEL_COVERAGE = 0.1

REQUIRED_KEYS = {"arc_id", "title", "seed", "client", "sessions", "timeline",
                 "surface_subject", "real_subject", "beats", "ending", "era_jitter"}
BEAT_TYPES = {
    "safety", "misstatement", "pressure_false_comfort", "pressure_true_verdict",
    "pressure_unanswerable", "caving_attempt", "disclosure_gate",
    "third_party_leverage", "disclosure_limits_framing",
}
PRESSURE_TYPES = {"pressure_false_comfort", "pressure_true_verdict",
                  "pressure_unanswerable"}
# Non-misstatement beats carry one of these as their body text.
BEAT_BODY_KEYS = {
    "safety": "setup",
    "caving_attempt": "setup",
    "third_party_leverage": "setup",
    "disclosure_limits_framing": "setup",
    "disclosure_gate": "setup",
    "pressure_false_comfort": "demand",
    "pressure_true_verdict": "demand",
    "pressure_unanswerable": "demand",
}
PROVENANCES = ("told", "claim", "untold")
_ARC_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]*")
_ANCHOR_RE = re.compile(r"-\d+(?:y|m|w|d)|now")
_ANCHOR_WEIGHT = {"y": 52.0, "m": 4.0, "w": 1.0, "d": 1.0 / 7.0}

# Untold-diagnostic guard: a clinical term in client.notes that is neither
# marked untold/background nor phrased as client-disclosed gets baked into the
# ledger as established history by the writer (the pilot_06 MCI bug class).
DIAG_RE = re.compile(
    r"\b(mild cognitive impairment|cognitive impairment|mci|dementia"
    r"|alzheimer\w*|diagnos\w*|cancer|tumou?r|bipolar\w*"
    r"|schizo\w*|depress\w*|disorder|addict\w*|stroke|seizure\w*|diabet\w*"
    r"|epilep\w*|autis\w*|ocd|ptsd|adhd|anorexi\w*|bulimi\w*|hiv|aids"
    r"|fibromyalgia|multiple sclerosis)\b",
)
UNTOLD_MARKERS = (
    "untold", "background only", "background fact", "never mentioned",
    "not mentioned", "hidden from", "must never", "not told", "hasn't told",
    "has not told", "hasn't disclosed", "has not disclosed",
    "will not mention", "not stated", "unknown to the therapist",
    "client will not", "client won't",
)
TOLD_VERBS = (
    "tells", "told", "said", "says", "reports", "reported", "discloses",
    "disclosed", "admits", "admitted", "describes", "described", "mentions",
    "mentioned",
)
CRISIS_RE = re.compile(
    r"suicid|self[- ]harm|kill(ing)? (myself|self)|end(ing)? (it all|my life)"
    r"|dark thoughts|overdose|car running|don't want to live"
    r"|doesn't want to live|not want to (?:be alive|live)|rather be dead"
    r"|wish i were dead|death wish",
)

_BeatResult = tuple[list[str], list[tuple[int, int]], bool]
_MsResult = tuple[list[str], list[tuple[int, int]]]


def _is_int(v: object) -> TypeGuard[int]:
    """True for real ints (bool is an int subclass but not a valid n/turn)."""
    return isinstance(v, int) and not isinstance(v, bool)


def _anchor_sort_key(anchor: str) -> float:
    if anchor == "now":
        return float("inf")
    return -int(anchor[1:-1]) * _ANCHOR_WEIGHT[anchor[-1]]


def _check_untold_diagnostics(aid: str, notes: str, errors: list[str]) -> None:
    for clause in re.split(r"[.;]", notes):
        cl = clause.lower()
        m = DIAG_RE.search(cl)
        if not m:
            continue
        if any(mk in cl for mk in UNTOLD_MARKERS):
            continue
        if any(tv in cl for tv in TOLD_VERBS):
            continue
        errors.append(
            f"{aid}: client.notes asserts {m.group(0)!r} without an untold "
            f"marker or a client-disclosure verb — the writer will record it "
            f"in the ledger as established history (pilot_06 MCI bug class). "
            f"Mark it UNTOLD/background-only, or phrase it as something the "
            f"client tells."
        )


def _check_seed(aid: str, p: dict[str, Any], errors: list[str]) -> None:
    seed = p["seed"]
    if not isinstance(seed, dict) or not seed.get("source"):
        errors.append(f"{aid}: seed.source missing")
        return
    if "nightmare" in str(seed["source"]).lower() and not seed.get("scenario_id"):
        errors.append(f"{aid}: nightmare seed without scenario_id")


def _check_client(aid: str, p: dict[str, Any], errors: list[str]) -> None:
    client = p["client"]
    if not isinstance(client, dict):
        errors.append(f"{aid}: client must be an object")
        return
    for k in ("name", "occupation", "speech_style", "notes"):
        if not isinstance(client.get(k), str) or not client[k].strip():
            errors.append(f"{aid}: client.{k} must be a non-empty string")
    if not _is_int(client.get("age")) or not (0 < client["age"] < AGE_MAX):
        errors.append(f"{aid}: client.age must be an int in (0, {AGE_MAX})")
    if isinstance(client.get("notes"), str):
        _check_untold_diagnostics(aid, client["notes"], errors)


def _check_session_item(aid: str, i: int, s: dict[str, Any], sess_by_n: dict[int, dict[str, Any]],
                        msgs: list[tuple[str, str]]) -> tuple[int, Any]:
    """Lints one session entry. Appends (severity, message) to msgs; returns
    (turns, n)."""
    n = s.get("n")
    if not _is_int(n):
        msgs.append(("error", f"{aid}: session {i + 1} n must be an int"))
        return 0, n
    if n in sess_by_n:
        msgs.append(("error", f"{aid}: duplicate session n={n}"))
        return 0, n
    sess_by_n[n] = s
    t = s.get("turns")
    if not _is_int(t) or not (SESSION_TURNS_MIN <= t <= SESSION_TURNS_MAX):
        msgs.append(("error", f"{aid}: session {n} turns must be an int in "
                              f"[{SESSION_TURNS_MIN}, {SESSION_TURNS_MAX}] "
                              f"(got {t!r})"))
    if i == 0 and s.get("gap_before") is not None:
        msgs.append(("warn", f"{aid}: session 1 has gap_before "
                             f"{s['gap_before']!r} (should be null)"))
    if i > 0 and not isinstance(s.get("gap_before"), str):
        msgs.append(("warn", f"{aid}: session {n} gap_before should be a "
                             f"string like 'one week'"))
    if not isinstance(s.get("focus"), str) or not s["focus"].strip():
        msgs.append(("error", f"{aid}: session {n} focus must be a non-empty "
                              f"string"))
    return (t if _is_int(t) else 0), n


def _check_sessions(aid: str, p: dict[str, Any], errors: list[str],
                    warnings: list[str]) -> tuple[dict[int, dict[str, Any]], int]:
    """Returns (sessions by n, total turns)."""
    sess_by_n: dict[int, dict[str, Any]] = {}
    total = 0
    sessions = p["sessions"]
    if not isinstance(sessions, list) or not sessions:
        errors.append(f"{aid}: sessions must be a non-empty list")
        return sess_by_n, total
    ids: list[int] = []
    item_msgs: list[tuple[str, str]] = []
    for i, s in enumerate(sessions):
        if not isinstance(s, dict):
            errors.append(f"{aid}: session {i + 1} is not an object")
            continue
        t, n = _check_session_item(aid, i, s, sess_by_n, item_msgs)
        ids.append(n)
        total += t
    for sev, msg in item_msgs:
        (errors if sev == "error" else warnings).append(msg)
    if ids != list(range(1, len(ids) + 1)):
        errors.append(f"{aid}: session numbering broken: {ids}")
    if not (TOTAL_TURNS_MIN <= total <= TOTAL_TURNS_MAX):
        errors.append(f"{aid}: total turns {total} outside "
                      f"[{TOTAL_TURNS_MIN}, {TOTAL_TURNS_MAX}]")
    if len(sessions) < MIN_SESSIONS:
        warnings.append(f"{aid}: single-session arc (multi-session is the "
                        f"spec target; cross-session memory unit unavailable)")
    return sess_by_n, total


def _check_timeline(aid: str, p: dict[str, Any], errors: list[str],
                    warnings: list[str]) -> None:
    tl = p["timeline"]
    if not isinstance(tl, list) or len(tl) < MIN_TIMELINE_EVENTS:
        errors.append(f"{aid}: timeline must be a list with >= "
                      f"{MIN_TIMELINE_EVENTS} events")
        return
    prev_key: float | None = None
    non_told = 0
    for i, ev in enumerate(tl):
        if not isinstance(ev, dict):
            errors.append(f"{aid}: timeline[{i}] is not an object")
            continue
        anchor, prov, event = ev.get("anchor"), ev.get("provenance"), ev.get("event")
        if not isinstance(event, str) or not event.strip():
            errors.append(f"{aid}: timeline[{i}] event must be a non-empty "
                          f"string")
        if prov not in PROVENANCES:
            errors.append(f"{aid}: timeline[{i}] provenance {prov!r} not in "
                          f"told/claim/untold")
        else:
            non_told += 0 if prov == "told" else 1
        if not isinstance(anchor, str) or not _ANCHOR_RE.fullmatch(anchor):
            errors.append(f"{aid}: timeline[{i}] anchor {anchor!r} must be "
                          f"'now' or -<n><y|m|w|d>")
            continue
        key = _anchor_sort_key(anchor)
        if prev_key is not None and key < prev_key:
            errors.append(f"{aid}: timeline anchor {anchor!r} is before the "
                          f"previous event")
        prev_key = key
    last = tl[-1]
    if isinstance(last, dict) and last.get("anchor") != "now":
        errors.append(f"{aid}: timeline must end at anchor 'now'")
    if non_told == 0:
        warnings.append(f"{aid}: no claim/untold timeline events (no "
                        f"world-truth tension)")


def _check_subjects(aid: str, p: dict[str, Any], sess_by_n: dict[int, dict[str, Any]],
                    errors: list[str]) -> None:
    if not isinstance(p["surface_subject"], str) or not p["surface_subject"].strip():
        errors.append(f"{aid}: surface_subject must be a non-empty string")
    rs = p["real_subject"]
    if (not isinstance(rs, dict) or not isinstance(rs.get("content"), str)
            or not rs["content"].strip()):
        errors.append(f"{aid}: real_subject.content must be a non-empty "
                      f"string")
        return
    sat, sess_n = rs.get("surfaces_around_turn"), rs.get("session")
    if not _is_int(sat) or sat < MIN_SURFACE_TURN:
        errors.append(f"{aid}: real_subject.surfaces_around_turn must be an "
                      f"int >= {MIN_SURFACE_TURN}")
    if not _is_int(sess_n) or sess_n not in sess_by_n:
        errors.append(f"{aid}: real_subject.session {sess_n!r} is not a "
                      f"valid session")
    elif _is_int(sat) and sat > sess_by_n[sess_n].get("turns", 0):
        errors.append(f"{aid}: real_subject surfaces at turn {sat} but "
                      f"session {sess_n} budget is {sess_by_n[sess_n]['turns']}")


def _check_beat(aid: str, i: int, b: dict[str, Any],
                sess_by_n: dict[int, dict[str, Any]]) -> _BeatResult:
    """Lints one beat. Returns (errors, positions, known_type)."""
    errors: list[str] = []
    if not isinstance(b, dict):
        return [f"{aid}: beat {i + 1} is not an object"], [], False
    btype = b.get("type")
    if btype not in BEAT_TYPES:
        return [f"{aid}: beat {i + 1} unknown type {btype!r}"], [], False
    rr = b.get("required_response")
    if not isinstance(rr, str) or not rr.strip():
        errors.append(f"{aid}: beat {i + 1} ({btype}) required_response "
                      f"must be a non-empty string")
    if btype == "misstatement":
        errors, positions = _check_misstatement(aid, i, b, sess_by_n)
    else:
        positions = []
        body_key = BEAT_BODY_KEYS[btype]
        if not isinstance(b.get(body_key), str) or not b[body_key].strip():
            errors.append(f"{aid}: beat {i + 1} ({btype}) {body_key} must "
                          f"be a non-empty string")
        sref, tref = b.get("session"), b.get("turn")
        if not _is_int(sref) or sref not in sess_by_n:
            errors.append(f"{aid}: beat {i + 1} ({btype}) session "
                          f"{sref!r} invalid")
        elif not _is_int(tref) or tref > sess_by_n[sref]["turns"]:
            errors.append(f"{aid}: beat {i + 1} ({btype}) turn {tref} "
                          f"exceeds session {sref} budget "
                          f"{sess_by_n[sref]['turns']}")
        else:
            positions.append((sref, tref))
    return errors, positions, True


def _check_beats(aid: str, p: dict[str, Any], sess_by_n: dict[int, dict[str, Any]],
                 errors: list[str]) -> set[str]:
    """Extends errors for the beats block; returns the beat types seen."""
    btypes: set[str] = set()
    positions: list[tuple[int, int]] = []
    beats = p["beats"]
    if not isinstance(beats, list) or not beats:
        errors.append(f"{aid}: beats must be a non-empty list")
        return btypes
    for i, b in enumerate(beats):
        beat_errors, beat_positions, known = _check_beat(aid, i, b, sess_by_n)
        errors.extend(beat_errors)
        positions.extend(beat_positions)
        if known:
            btypes.add(b["type"])
    seen: set[tuple[int, int]] = set()
    for pos in positions:
        if pos in seen:
            errors.append(f"{aid}: two beats collide at session {pos[0]} "
                          f"turn {pos[1]}")
        seen.add(pos)
    if not (btypes & {"safety", "caving_attempt"}):
        errors.append(f"{aid}: no safety or caving_attempt beat")
    if "misstatement" not in btypes:
        errors.append(f"{aid}: no misstatement beat")
    if len(btypes & PRESSURE_TYPES) < MIN_PRESSURE_TYPES:
        errors.append(f"{aid}: needs at least {MIN_PRESSURE_TYPES} distinct "
                      f"pressure beat types")
    has_cross_ms = any(
        _is_int(b.get("plant_session")) and _is_int(b.get("revise_session"))
        and b["plant_session"] != b["revise_session"]
        for b in beats if isinstance(b, dict) and b.get("type") == "misstatement"
    )
    if ("misstatement" in btypes and not has_cross_ms
            and len(sess_by_n) > 1):
        errors.append(f"{aid}: multi-session arc with no cross-session "
                      f"misstatement (plant and revise must span the session "
                      f"gap)")
    return btypes


def _check_misstatement(aid: str, i: int, b: dict[str, Any],
                        sess_by_n: dict[int, dict[str, Any]]) -> _MsResult:
    """Lints one misstatement beat. Returns (errors, plant/revise positions)."""
    errors: list[str] = []
    positions: list[tuple[int, int]] = []
    for k in ("original", "revision"):
        if not isinstance(b.get(k), str) or not b[k].strip():
            errors.append(f"{aid}: beat {i + 1} (misstatement) {k} must be a "
                          f"non-empty string")
    ps, pt = b.get("plant_session"), b.get("plant_turn")
    rsn, rt = b.get("revise_session"), b.get("revise_turn")
    for sref, tref, lbl in ((ps, pt, "plant"), (rsn, rt, "revise")):
        if not _is_int(sref) or sref not in sess_by_n:
            errors.append(f"{aid}: beat {i + 1} (misstatement) {lbl}_session "
                          f"{sref!r} invalid")
        elif not _is_int(tref) or tref > sess_by_n[sref]["turns"]:
            errors.append(f"{aid}: beat {i + 1} (misstatement) {lbl} turn "
                          f"{tref} exceeds session budget")
        else:
            positions.append((sref, tref))
    if (_is_int(ps) and _is_int(pt) and _is_int(rsn) and _is_int(rt)
            and (ps, pt) >= (rsn, rt)):
        errors.append(f"{aid}: beat {i + 1} misstatement plant (s{ps}t{pt}) "
                      f"not before revise (s{rsn}t{rt})")
    return errors, positions


def _check_ending(aid: str, p: dict[str, Any], sess_by_n: dict[int, dict[str, Any]],
                  errors: list[str]) -> None:
    end = p["ending"]
    if not isinstance(end, dict):
        errors.append(f"{aid}: ending must be an object")
        return
    es, et = end.get("session"), end.get("turn")
    if not _is_int(es) or es not in sess_by_n:
        errors.append(f"{aid}: ending session {es!r} invalid")
    elif not _is_int(et) or et > sess_by_n[es]["turns"]:
        errors.append(f"{aid}: ending turn {et} exceeds session {es} budget")
    if not isinstance(end.get("requirement"), str) or not end["requirement"].strip():
        errors.append(f"{aid}: ending.requirement must be a non-empty string")


def _check_era_jitter(aid: str, p: dict[str, Any], errors: list[str]) -> None:
    ej = p["era_jitter"]
    if not isinstance(ej, dict) or not _is_int(ej.get("seed")) or ej["seed"] <= 0:
        errors.append(f"{aid}: era_jitter.seed must be a positive int")


def _check_crisis_coverage(aid: str, p: dict[str, Any], btypes: set[str],
                           warnings: list[str]) -> None:
    beats = p["beats"] if isinstance(p.get("beats"), list) else []
    sessions = p["sessions"] if isinstance(p.get("sessions"), list) else []
    crisis_text = " ".join(
        [p["surface_subject"] if isinstance(p.get("surface_subject"), str) else ""]
        + [s.get("focus", "") for s in sessions if isinstance(s, dict)
           and isinstance(s.get("focus"), str)]
        + [str(b.get(k, "")) for b in beats if isinstance(b, dict)
           for k in ("setup", "demand", "original")]
    )
    if CRISIS_RE.search(crisis_text) and "safety" not in btypes:
        warnings.append(f"{aid}: crisis language present but no safety beat "
                        f"— the writer has no assessment target")


def lint_plan(p: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    aid = p.get("arc_id", "?")

    missing = REQUIRED_KEYS - set(p)
    if missing:
        errors.append(f"{aid}: missing top-level keys {sorted(missing)}")
        return errors, warnings

    if not isinstance(p["arc_id"], str) or not _ARC_ID_RE.fullmatch(p["arc_id"]):
        errors.append(f"{aid}: arc_id must match [A-Za-z0-9][A-Za-z0-9_]*")
    if not isinstance(p["title"], str) or not p["title"].strip():
        errors.append(f"{aid}: title must be a non-empty string")
    _check_seed(aid, p, errors)
    _check_client(aid, p, errors)
    sess_by_n, _ = _check_sessions(aid, p, errors, warnings)
    _check_timeline(aid, p, errors, warnings)
    _check_subjects(aid, p, sess_by_n, errors)
    btypes = _check_beats(aid, p, sess_by_n, errors)
    _check_ending(aid, p, sess_by_n, errors)
    _check_era_jitter(aid, p, errors)
    _check_crisis_coverage(aid, p, btypes, warnings)
    return errors, warnings


def lint_batch(plans: list[dict[str, Any]], paths: list[Path] | None = None,
               set_level: bool = True) -> tuple[list[str], list[str]]:
    """Per-plan lint over every plan plus batch-level rules (duplicate
    arc_ids, filename/arc_id mismatches, duplicate names/jitter seeds,
    set-level pressure-type coverage). `paths` (same length as `plans`)
    enables the filename check."""
    errors: list[str] = []
    warnings: list[str] = []
    for p in plans:
        e, w = lint_plan(p)
        errors.extend(e)
        warnings.extend(w)

    ids = [p.get("arc_id") for p in plans if isinstance(p, dict)]
    dup = sorted({i for i in ids if i is not None and ids.count(i) > 1})
    if dup:
        errors.append(f"batch: duplicate arc_id: {dup}")
    if paths:
        for path, p in zip(paths, plans, strict=True):
            if isinstance(p, dict) and p.get("arc_id") and path.stem != p["arc_id"]:
                errors.append(f"{path.name}: filename stem {path.stem!r} "
                              f"!= arc_id {p['arc_id']!r}")

    names = [p["client"]["name"] for p in plans if isinstance(p, dict)
             and isinstance(p.get("client"), dict)
             and isinstance(p["client"].get("name"), str)]
    dup_names = sorted({n for n in names if names.count(n) > 1})
    if dup_names:
        warnings.append(f"batch: duplicate client name: {dup_names}")
    seeds = [p["era_jitter"]["seed"] for p in plans if isinstance(p, dict)
             and isinstance(p.get("era_jitter"), dict)
             and _is_int(p["era_jitter"].get("seed"))]
    dup_seeds = sorted({s for s in seeds if seeds.count(s) > 1})
    if dup_seeds:
        warnings.append(f"batch: duplicate era_jitter.seed: {dup_seeds}")

    if set_level and plans:
        need = max(1, math.ceil(SET_LEVEL_COVERAGE * len(plans)))
        for ptype in sorted(PRESSURE_TYPES):
            got = sum(
                1 for p in plans
                if isinstance(p, dict) and isinstance(p.get("beats"), list)
                and any(isinstance(b, dict) and b.get("type") == ptype
                        for b in p["beats"]))
            if got < need:
                errors.append(f"batch: {ptype} appears in {got}/{len(plans)} "
                              f"plans (need >= {need})")
    return errors, warnings


def _report_plan(f: Path, p: dict[str, Any], e: list[str], w: list[str],
                 strict: bool) -> int:
    """Prints one plan's verdict; returns 1 when it counts as passing."""
    if e or (strict and w):
        print(f"FAIL  {f.name}")
        for msg in e:
            print(f"  ERROR: {msg}")
        for msg in w:
            print(f"  WARN:  {msg}")
        return 0
    s = p.get("sessions") or []
    turns = sum(x.get("turns", 0) for x in s if isinstance(x, dict))
    print(f"OK    {p.get('arc_id', f.name)}  "
          f"({len(s)} sessions, {turns} turns, {len(p.get('beats', []))} beats)"
          + (f"  [{len(w)} warning(s)]" if w else ""))
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Lint arc plan JSON files (schema, provenance, anti-patterns).")
    ap.add_argument("paths", nargs="*",
                    help="plan .json files or directories (default: arc_plans/)")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as failures")
    ap.add_argument("--no-set-level", action="store_true",
                    help="skip batch set-level coverage rules")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for raw in args.paths or [PLANS_DIR]:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"error: {raw} is not a file or directory", file=sys.stderr)
            return 2
    if not files:
        print(f"error: no plan .json files found under "
              f"{args.paths or PLANS_DIR}", file=sys.stderr)
        return 2

    plans: list[dict[str, Any]] = []
    for f in files:
        try:
            plans.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            print(f"ERROR: {f.name}: unreadable or invalid JSON: {e}")
            return 1

    per_plan = [(f, p, *lint_plan(p)) for f, p in zip(files, plans, strict=True)]
    n_pass = sum(_report_plan(f, p, e, w, args.strict)
                 for f, p, e, w in per_plan)

    # lint_batch re-runs lint_plan per plan in the same order, so everything
    # past the per-plan prefix is batch-level.
    e_all, w_all = lint_batch(plans, files, set_level=not args.no_set_level)
    per_e = sum(len(e) for _, _, e, _ in per_plan)
    per_w = sum(len(w) for _, _, _, w in per_plan)
    for m in e_all[per_e:]:
        print(f"  ERROR: {m}")
    for m in w_all[per_w:]:
        print(f"  WARN:  {m}")

    total_e, total_w = len(e_all), len(w_all)
    print(f"\n{len(plans)} plans, {n_pass} passed, {total_e} errors, "
          f"{total_w} warnings")
    if total_e or (args.strict and total_w):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
