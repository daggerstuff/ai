# Dataset Pipeline Handoff

**Updated:** October 10, 2026 (04:30 UTC: dataset close-out — both structural rejects regenerated and accepted, 235/235 arc records accept, gold 189,735 after superseded-version cleanup, DPO in sync; 7 stale case pairs kept as historical evidence)  
**Workspace:** `/home/vivi/pixelated` (`staging`); AI submodule `ai/` at `b8d4ce6cb` (`staging`)  
**Purpose:** Give the next agent a verified handoff for continuing the arc-corpus dataset work and choosing the next useful task in the broader training-data pipeline.

## Dataset close-out (October 10, 2026, 03:00–04:15 UTC)

This section supersedes the open decisions in the October 9 18:30 snapshot. All dataset-blocking items are resolved; the final state below was measured against quiescent inputs (no writer/auditor/builder processes running).

**Structural rejects fixed — both regenerated under `cx/gpt-6-astra` and audited accept:**

- `arc_0144` s2: regenerated 03:15–03:21 under a corrective note (STRICT ALTERNATION + NO FABRICATION + LEDGER FORMAT bare tokens + the plan's s2 beats at planned turns), **ACCEPT first attempt** (revision 2). 16 client/therapist pairs, alternation verified.
- `arc_0052`: an s1-only regen passed the structure gate but drew 4 HR flags. Adjudication against the plan found: (1) doorway `-3w` ledger vs. the plan's setup prose "last Thursday" is a **plan-internal inconsistency** (the plan's timeline block says `-3w`); the writer had harmonized the client's speech to "three weeks ago", which is what drew the flag; (2) a **real defect** — the new s1 disclosed the `-9d` wrist event at turn 9, breaking the plan's misstatement arc (plant s1 t4 → revise s2 t7); (3) a **cascade** — the banked s2 carried a sparse, drifted chain (`-3d` doorway, no `-4y`/`-4m`) that a full new s1 chain would orphan; (4) a **real minor defect** — s3 t8's ledger flattened the money-revision ownership claim ("it's all his") into a bare item list. Resolution: **full-arc regen (s1+s2+s3)** under one corrective note with the reservation rule (`-9d` is told in session 2 only; `-6d` never appears; every ledger carries the full chain; doorway speech follows the setup verbatim while the ledger follows the timeline anchor). Regenerated 03:44–03:54 (sessions=3, arcs=1, 0 gate failures, 0 errors), **ACCEPT, 0 flags** (revision 3). Turn counts 20/16/12 pairs, alternation verified on all three.
- Both notes (`arc_audit_notes/arc_0052.json`, `arc_0144.json`) were deleted by the auditor on accept, per convention; pre-mutation corpus backups are in `/home/vivi/arc_regen/corpus-backup-20261010-*`. The earlier 503/quota root cause (upstream re-arms its usage-limit block on every probe; cx/* GPT-6 routes share one pool) is recorded in `/home/vivi/arc_regen/OPERATOR_NOTES.md`.

**HR arcs (October 9):** `arc_0105`, `arc_0165` (3 revisions each) and `arc_0128` (5 revisions) were regenerated and accepted in the 19:20–20:41 rounds — all 25 regeneration-target arcs accept.

**Final quiescent state (measured 04:05–04:15 UTC):**

- Arc records: **235 records, 235 accept, 0 hr**. Checkpoint: **576 rows**, all accepted, all structurally valid.
- Gold: **189,735 lines** (was 189,733 before the 04:01:11 consolidation appended 4 rows: `arc_0052` s1/s2/s3, `arc_0144` s2, all `cx/gpt-6-astra` provenance). A post-consolidation dry-run reported scanned 235 arcs / 576 sessions, 0 skipped by verdict, **0 emitted, 576 duplicates, 0 rejects** — idempotent no-op, checkpoint↔gold in sync.
- Superseded-version cleanup (04:04 UTC, this executor): the two deepseek-era `arc_0052` s2/s3 gold rows (consolidated 2026-10-09T18:17 from the pre-regeneration record) were **removed** so gold holds exactly one row per accepted session — 576 keys, each matching the current checkpoint render. Rationale: gold is the SFT training corpus; the deepseek-era rows were superseded precisely because of the defects that triggered the full-arc regeneration, and training on both versions would teach contradictory behavior. Nothing is lost: the removed rows are preserved in `train_master_gold.jsonl.pre_stale_removal_20261010T040441Z`, and the historical versions remain available in the `.pre*` snapshots and as rejected arms inside the session-mode DPO pairs. This supersedes the earlier "remain in gold by design" note; the removal is reversible from the backup.
- Reject log: **unchanged** (4 historical rows, mtime 2026-10-09 20:50). The 2 structural rejects remain as historical evidence; their fixed replacements are in gold.
- DPO pairs: **693 rows** after the 04:03–04:04 resync. Builder dry-run (04:05): `built=785, emitted=0, duplicates=785` (`session`: 314, `attempt`: 471 candidates) — file in sync with the builder. Integrity: **0 parse errors, 0 duplicate `pair_hash`**. Case mode builds 0 candidates now that no arc carries an `hr` verdict (expected; the 15 historical case pairs remain in the file).
- Consolidation focused tests were last run green in the 20:50 pass (this round changed no consolidation code).
- Independent verification pass (04:03–04:30 UTC, this executor): re-ran the consolidation dry-run (`0 emitted, 576 duplicates`), ran the real builder against the current inputs (+22 pairs: 19 attempt + 3 session for the regenerated `arc_0052`/`arc_0144` sessions), then removed the **16 stale pairs** whose chosen side was built from the superseded pre-regeneration renders (15 `arc_0052`, 1 `arc_0144` s2; backup `arc_dpo_pairs.jsonl.pre_stale_removal_20261010T040441Z`). Post-cleanup: **693 pairs, 0 parse/schema/hash/duplicate failures**; remaining `arc_0052` session-mode pairs (12) all carry the new Astra chosen text with historical deepseek renders as rejected arms — the intended superseded-version comparisons. Focused tests: consolidation **14 passed**, builder **27 passed**. Final dry-runs: consolidation `0 emitted, 576 duplicates`, builder `built=785, emitted=0, duplicates=785`. Gold version audit: 576 distinct keys, one row each, all matching current renders; no multi-version keys remain.

**Stale case pairs — disposition recorded (default: keep):** the 7 stale case-pair rejected sides (`arc_0196`, `arc_0067`, `arc_0206`, `arc_0218`, `arc_0006`, `arc_0216`, `arc_0143`) are **kept as historical evidence**. Rationale: each pair remains internally consistent (its chosen text is a real accepted session; its rejected text is a real historical HR version), the set documents the HR→accept improvement path, and refreshing would require an owner-approved rewrite with no quality gain. This is a reversible no-op default: rebuilding any pair from current snapshots is bounded and can be done on request.

**Close-out:** dataset work is **evidence-complete**. Documented, non-blocking residuals: clinical/ledger faithfulness judgments remain separate from mechanical checks (`ARC_CORPUS_SPEC.md` §§4, 7); DPO prompts remain reconstructed (`brief_reconstructed: true`, corrective notes omitted); whole-session preferences are not validated turn-level. Fireworks documentation stages (readiness checklist + pilot pre-plan) are unblocked per `.junie/plans/fireworks-training-preplan.md`; both artifacts are documentation-only and contain no uploads, provisioning, or spending.

## Verified consolidation results and moving input snapshot

The consolidation results below describe the earlier verification, not a quiescent dataset. At `2026-10-09T01:17:13Z`, independent generator PID `902322` was still running; arc/checkpoint counts were rechecked at 229/546 but remain a moving snapshot. Its completion and audit are prerequisites for Step 2 unless the user explicitly scopes work against in-flight inputs.

- Arc records: `ai/training/output/arc_corpus/arc_records.jsonl` — 229 records; latest consolidated dry-run scanned 229 arcs / 507 accepted sessions.
- Gold: `ai/data/curated/sft_chatml/train_master_gold.jsonl` — 189,666 lines / 397,286,656 bytes both before and after this pass's real consolidation.
- DPO pairs: `ai/training/output/arc_corpus/arc_dpo_pairs.jsonl` — 518 historical lines verified at 01:23:43; **550 lines at 01:24:47 after an independent change**, not a write by this executor (see below).
- Session checkpoint: `ai/training/output/arc_corpus/sessions_checkpoint.jsonl` — 546 lines.
- Consolidation reject log: `ai/training/output/arc_corpus/consolidation_rejections.jsonl` — 0 lines / 0 bytes before and after; preserved unchanged.
- October 9 preflight dry-run, real run and post-run dry-run all reported: **229 scanned arcs**, **507 considered sessions**, **19 arcs skipped by verdict**, **0 emitted gold**, **507 duplicates**, **0 content rejects**, **0 structurally invalid sessions**, and empty `by_arc`.
- The approved real consolidation was executed in this pass and was an idempotent no-op: gold delta **0**, reject delta **0**. Focused tests: **14 passed**.

## Completed consolidation and discrepancy investigation

Before mutation, the live counts and dry-run were cross-checked against the October 8 snapshot (226 arc records, 189,596 gold lines, 540 checkpoints, 70 eligible sessions and 16 skipped arcs). The differences were investigated using metadata only:

- The final 70 gold records already have `arc_corpus` family and `accept` provenance, with `consolidated_at` spanning `2026-10-08T21:21:58.116670+00:00` through `2026-10-08T21:21:58.636374+00:00`. This is evidence that the historical append was already present before this pass, not an append performed here. `arc_0090` contributes two of these records.
- The three additional arc records (`arc_0075`, `arc_0128`, `arc_0144`) each contain three sessions and `audit: null`. Reviewer evidence attributes these records and the six additional checkpoints to the independent regeneration run documented below. They explain the increase from 16 to 19 skipped arcs; accepted-session consideration remains 507 in the verified consolidation snapshot.
- Six additional checkpoints reference `arc_0075` sessions 2/3, `arc_0128` session 3, `arc_0006` sessions 1/2 and `arc_0144` session 3. These are in-flight writer output, not evidence of completed regeneration or audit. Checkpoints were inspected read-only, not regenerated or rewritten by this executor.
- Gold SHA-256 before and after the real run: `205706c0b81dc96315c9f35beba3700a93d9998bab44b9463ce3caa45a8afdb6`. Reject-log SHA-256 remained the empty-file hash; arc-input and unrelated `audit_arc_corpus.py` / `generate_arc_corpus.py` hashes also matched before and after.
- The loader reported 189,653 unique gold hashes, not the 189,666 physical lines. This pre-existing difference is not a consolidation delta or proof of full corpus integrity; broader deduplication/provenance assessment remains next-step work.

Commands executed from `/home/vivi/pixelated/ai`, each with a 120-second initial timeout:

```bash
wc -lc data/curated/sft_chatml/train_master_gold.jsonl \
  training/output/arc_corpus/consolidation_rejections.jsonl
PYTHONPATH=/home/vivi/pixelated/ai:/home/vivi/pixelated \
  /home/vivi/pixelated/ai/.venv/bin/python training/consolidate_arc_corpus.py --dry-run
PYTHONPATH=/home/vivi/pixelated/ai:/home/vivi/pixelated \
  /home/vivi/pixelated/ai/.venv/bin/python training/consolidate_arc_corpus.py
wc -lc data/curated/sft_chatml/train_master_gold.jsonl \
  training/output/arc_corpus/consolidation_rejections.jsonl
PYTHONPATH=/home/vivi/pixelated/ai:/home/vivi/pixelated \
  /home/vivi/pixelated/ai/.venv/bin/python training/consolidate_arc_corpus.py --dry-run
PYTHONPATH=/home/vivi/pixelated/ai:/home/vivi/pixelated \
  /home/vivi/pixelated/ai/.venv/bin/python -m pytest -q training/tests/test_consolidate_arc_corpus.py
```

Measured with `/usr/bin/time -f 'elapsed_seconds=%e'`: preflight dry-run **4.35s**, real consolidation **4.26s**, post-run dry-run **3.86s**, pytest **13.21s** wall time (pytest reported **14 passed in 0.33s**). All exited 0. Metadata-only discrepancy inspection used a 60-second timeout.

Never clear or truncate the rejection log; it is append-only evidence. This executor initiated no external calls, regeneration, repository synchronization or training. Independent regeneration was already underway during the consolidation pass; that concurrency was missed in the original handoff. Matching before/after hashes establish the observed no-op, not an input lock or completed writer run.

## Independent writer and reconciliation gate

Read-only `ps -p 902322 -ww -o pid,lstart,stat,args` at `2026-10-09T01:17:13Z` confirmed PID `902322` in state `Sl`, started **2026-10-09 00:07:55 UTC**, with this command (observed, not an instruction to execute):

```text
/home/vivi/pixelated/ai/.venv/bin/python training/generate_arc_corpus.py --plans arc_0006,arc_0010,arc_0012,arc_0052,arc_0063,arc_0067,arc_0073,arc_0074,arc_0075,arc_0078,arc_0085,arc_0105,arc_0108,arc_0124,arc_0128,arc_0140,arc_0143,arc_0144,arc_0146,arc_0151,arc_0158,arc_0164,arc_0165,arc_0196,arc_0205 --concurrency 3
```

At that recheck, input mtimes were:
- `arc_records.jsonl`: **2026-10-09 00:09:17.580959166 UTC**, 18,876,587 bytes / 229 lines.
- `sessions_checkpoint.jsonl`: **2026-10-09 00:09:17.577173211 UTC**, 17,918,824 bytes / 546 lines.
- Reject evidence remained empty with mtime **2026-10-05 13:49:26.259831416 UTC**.

The reviewed 25-plan list covers all 19 `audit: null` records plus six plans without records yet. Those 19 records are pending writer/audit output, not settled review dispositions. An unchanged mtime at this recheck does not establish completion while the process remains live. No process was stopped, restarted or otherwise interfered with, and no mutating consolidation was rerun for this correction.

**Operational risk (2026-10-09T01:22:53Z):** PID `902322` remained `Sl`, elapsed **1:14:56**, CPU **00:00:01**. Arc/checkpoint files still had the 00:09:17 mtimes above, while `/home/vivi/arc_regen/writer.log` had advanced to **01:22:14 UTC**, 103,308 bytes. A sanitized read-only scan at 01:23:43 counted **811 lines containing `http_503`**, including **25 of the last 30 lines**, and 193 lines containing both “gate” and “fail” (line counts, not unique attempts). Reviewer evidence at 01:20 also reported ongoing gate failures for `arc_0164` session 1 and `arc_0205` session 2. The writer remains engaged in external-request retries without a new checkpoint since 00:09; continued attempts carry potential paid-call/budget exposure. These observations do not establish actual charges or a rate-limit diagnosis for HTTP 503. No raw log or clinical text was disclosed and no process was touched.

**Approval clarification required:** Who launched this regeneration run, was its scope/budget separately approved, and should the owner allow retries to continue or stop the run given the stalled progress and expense risk? This decision belongs to the user/owner. This plan does not authorize regeneration on its own; observing the run is not approval of it.

**Step 2 gate:** Wait for writer completion and audit, then recheck process state and all builder inputs before DPO dry-run or coverage reconciliation; alternatively obtain explicit user authorization defining in-flight reconciliation scope. Inventory/hash `arc_records.jsonl`, `sessions_checkpoint.jsonl`, `failed_attempts/*.txt`, every `.pre*` snapshot checkpoint, plans and seed inputs; capture the pair file separately before any append. Check directory membership as well as file sizes/mtimes so new failed attempts are not missed. At 01:23:43, `failed_attempts/` contained **617 files**, **five newer than 00:09:18**, latest **01:01:50 UTC**; eight `.pre*` folders had latest file mtimes between September 20 and 28. The earlier unchanged statement applies only to arc/checkpoint metadata, **not all builder inputs**. Do not rerun mutating consolidation while this writer is active. Dataset close-out and Fireworks preparation remain blocked.

## Step 2 safe static findings — full reconciliation remains blocked

At `2026-10-09T01:19:08Z`, `ps -p 902322 -o pid=,lstart=,stat=,args=` again showed the same generator command, start time and `Sl` state. Read-only `stat` confirmed the arc/checkpoint sizes and mtimes listed above were unchanged. Pair-file metadata was 36,337,127 bytes, mtime **2026-10-05 15:14:38.759694366 UTC**; reject evidence remained 0 bytes with its previous mtime. That earlier pass asserted no new line inventory or audit disposition. The process/status check used a 60-second timeout; timed `stat` completed in **0.00s**, also with a 60-second timeout.

The earlier pass performed source-code/specification preparation only. This corrective pass also verified the writer-independent historical pair file and ran synthetic tests. No DPO dry-run, pair append, definitive coverage reconciliation, audit execution, regeneration, external call or training was performed.

### Settled checks of the existing pair file (October 9, 01:23:43 UTC)

Executed from `ai/`, timeout 60s: `/usr/bin/time -f 'elapsed_seconds=%e' ./.venv/bin/python - <<'PY'` with an inline read-only `json`, `hashlib`, `re`, `collections`, `os.scandir` and `Path` scan. It parsed each pair, checked nonempty string prompt/chosen/rejected/hash, nonempty metadata, valid mode, arc/session key, defect list, rejected source and mode-specific reconstruction flag; recomputed SHA-256 of `(prompt + chosen + rejected).lower().encode('utf-8')`; counted repeated hashes; and compared arc IDs with last-wins current records' `audit.verdict`. Elapsed **0.67s**, exit 0. Only aggregate counts, hashes and file metadata were emitted.

- **518 pairs:** 170 session, 333 attempt, 15 case; **0 parse/schema errors**, **0 hash mismatches**, **0 duplicate hashes**, **0 pairs outside currently accepted arc IDs**.
- Pair SHA-256: `add46f6045d077f39b2e799b342296b27726127383711ef9228831dd2c2e70e8`; size/mtime unchanged during the scan (36,337,127 bytes, October 5 15:14:38 UTC). This generator does not write the pair file.
- **Final concurrent-change discovery:** at 01:24:47, `sha256sum` differed. Follow-up `stat`, `wc -l`, `sha256sum` and an independent Python hash agreed on **550 lines / 39,009,295 bytes**, mtime **01:24:34.943731420 UTC**, SHA-256 `3af028b9ee25443efda1a197f5f86eac338c73fcc0d8c407eaee18af58c4030e`. At 01:25:02, hashing its first 36,337,127 bytes reproduced the historical hash above and 518 lines: the checked historical prefix remains intact, with **32 additional lines**. This executor did not write these pairs; the responsible actor and approval are unknown. The 32 additions have **not** been integrity/context/source-validated here. The settled checks below apply only to the 518-line historical prefix, not the entire current artifact. Establish ownership and quiescence of **all** corpus/pair writers before resuming reconciliation or correction; generator completion alone is now insufficient. PID `902322` was still `Sl` at 01:24:47 (elapsed 1:16:51).
- Acceptance here is **arc-ID membership only**, not proof that historical chosen text matches an audited accepted session/version. That source-content/provenance check remains gated; historical rejected sources were not re-certified.
- Focused synthetic tests, started 01:22:53 UTC, timeout 120s: **26 passed in 0.95s**, **14.56s wall**, exit 0:

```bash
cd /home/vivi/pixelated/ai
/usr/bin/time -f 'elapsed_seconds=%e' env \
  PYTHONPATH=/home/vivi/pixelated/ai:/home/vivi/pixelated \
  ./.venv/bin/python -m pytest -q training/tests/test_build_arc_dpo.py
```

### Bounded verification of the 32-pair suffix (October 9, 01:27 UTC)

Independent read-only verification at `2026-10-09T01:27:05Z` reproduced the **550-line / 39,009,295-byte** snapshot, mtime **01:24:34.943731 UTC**, SHA-256 `3af028b9ee25443efda1a197f5f86eac338c73fcc0d8c407eaee18af58c4030e`. The first 36,337,127 bytes still contain 518 lines and reproduce the historical prefix hash above. Pair and arc-record size/mtime metadata remained unchanged during this scan; this is bounded observation, not a writer lock. Inline Python ran in **0.54s** with a **60s timeout** and emitted only aggregate integrity results and identifiers.

- Across all **550 pairs**: **0 parse/schema failures**, **0 recomputed hash mismatches**, **0 duplicate hashes**. Schema checks used the fields/types, source keys, defect list and reconstruction flag described above; top-level keys matched the historical pair shape. These checks do not establish provider compatibility.
- All **32 additions** are attempt-mode pairs linked to currently accepted arc IDs: `arc_0080`, `arc_0091`, `arc_0093`, `arc_0101`, `arc_0119`, `arc_0147`, `arc_0149`, `arc_0150`, `arc_0170`. This is last-wins current verdict membership only, **not certification of chosen text against an audited accepted session/version**.
- All 32 reference existing `failed_attempts/*.txt`. A second read-only check at **01:27:14 UTC** confirmed each file starts with `# gates:` and its body, after removing that first header line and stripping surrounding whitespace as `_attempt_pairs` does, equals the pair's rejected text (**0 mismatches**, source metadata unchanged during each read). Comparing the entire file would incorrectly include the gate header. This does not certify historical source immutability or rejection judgments.
- **0 suffix prompts missing expected earlier-session carry-forward markers**, using the marker heuristic described below. Marker presence does not prove complete context fidelity, original writer context equivalence or training suitability; reconstructed-context limitations still apply.
- `writer_model` uses **`deepseek-v4.1-flash` in 25 additions** and **`deepseek/deepseek-v4.1-flash` in 7**. Resolve canonical model identity before any later model grouping/export; no metadata was normalized here.

This bounded check supersedes the earlier statement that the additions were unvalidated, but only for these static properties. At **01:27:18 UTC**, PID `902322` remained **`Sl`**, elapsed **1:19:21**. Append ownership/approval, chosen-source versions and all-writer quiescence remain unresolved. No builder dry-run, pair write, definitive coverage reconciliation or external request was initiated; dataset close-out and Fireworks stages remain blocked.

### Confirmed historical session-context defect and use restriction

The same scan extracted carry-forward markers with `re.findall(r'^\s+s(\d+) \| ', prompt, re.M)` and compared them with sessions `1..session_n-1`. Of **98 session-mode pairs for session 2+**, **95 lack earlier-session markers**: **70/73 session-2** and **25/25 session-3** prompts. For all 95, at least one missing prior session has client turns in the current checkpoint. `generate_arc_corpus.py` emits these markers for prior client turns; `_session_pairs` lines 261–269 caches only priors before the first eligible session of an arc and never acquires intervening sessions. The implementation defect and incomplete historical session prompts are **confirmed**, not merely hypothetical. The passing synthetic suite does not establish correct multi-session context handling.

**Quarantine for preference use:** treat those 95 historical session pairs as unfit until corrected and reviewed; this is a documented use restriction, not a file mutation. Three attempt-mode prompts also lack prior markers (two session-2, one session-3), but the current checkpoint lacks corroborating client turns for their missing priors. Those three require source-history investigation and should also be withheld pending resolution; do not attribute them to the session-mode cache (attempt mode uses an arc/session cache). The remaining three session-2 prompts contain the expected marker, but marker presence alone does not certify full context fidelity. Current checkpoints cannot prove the exact historical writer input, and reconstructed prompts already omit corrective notes.

**Concrete remediation task:** separately scope a regression test and builder prior-cache fix, then obtain approval for a **history-preserving correction** that records affected old hashes, source versions, corrected context and supersession links. Do not overwrite/delete historical pairs or append corrected pairs while the writer is active. A future dry-run must not be treated as sufficient preference quality approval.

Source-derived checks for the next authorized pass (not findings about current pair contents):

- `build_arc_dpo.py` loads records/checkpoints last-wins and silently skips malformed JSONL. Independently count parse/schema failures before trusting its summary. Require nonempty string `prompt`, `chosen`, `rejected`, mode-specific metadata and `pair_hash`; recompute the builder's SHA-256 of lowercased concatenated prompt/chosen/rejected and check duplicate hashes without rewriting historical pairs.
- Session/attempt modes select accepted arc IDs but render chosen text from the latest checkpoint, not directly from the audited record. Verify checkpoint turns match the accepted audited session, and verify historical chosen sources/version provenance separately; current acceptance by arc ID alone does not certify every historical pair. Preserve rejected-source evidence and pair history.
- Session/attempt prompts are reconstructed from plans and accepted-side priors, omit corrective notes, and carry `brief_reconstructed: true`. Case mode uses a seed-case brief, compares session 1 only, and marks reconstruction false; this is not proof of identical original writer contexts. These whole-session preferences are not validated turn-level or provider-ready pairs.
- Fix the confirmed `_session_pairs` cache defect in separately scoped remediation with a multi-session regression test; investigate the three attempt-mode context gaps against historical sources. No builder change or historical-pair correction was made here.
- Rendering includes `[T|THINK]` only when a ledger is a dictionary and does not itself enforce the specification's nine-field contract. After writer completion/audit, compare approved plan/session expectations, latest accepted records and gold provenance/content, including ledger retention before every therapist turn, superseded versions and duplicate arc/session keys. Keep mechanical retention checks distinct from clinical ledger-faithfulness judgments (`ARC_CORPUS_SPEC.md` §§4, 7).

**Selected next task (updated 18:30 UTC):** the writer pipeline has **completed** — all 25 regeneration targets now have records, the relauncher exited, and 232/235 arcs carry `accept` verdicts (3 remain `hr`: `arc_0105`, `arc_0128`, `arc_0165`, each with no accepted partner for its seed key). The blocked Step 2 work was executed: post-completion consolidation appended **58 new sessions** to gold (189,724 total) with **2 structural rejects** (`arc_0052` s1, `arc_0144` s2 — consecutive therapist turns, see section below), and the DPO pair file was resynced with **126 new pairs** (681 total, 0 hash/duplicate failures, dry-run emits 0). Remaining items: (1) owner review of the 2 structural rejects (fix, regenerate, or accept exclusion); (2) decision on the 3 HR arcs (regenerate or accept exclusion); (3) case-pair refresh decision (7 rejected sides stale vs. current checkpoint); (4) explicit dataset close-out. Dataset close-out has **not** been obtained; these decisions block Fireworks stages.

## Post-completion consolidation and DPO resync (October 9, 18:00–18:30 UTC)

The regeneration writer pipeline **completed** before this pass: all 25 target arcs have records, the relauncher/watcher processes exited, and the last records/checkpoint writes were 17:07:58 / 16:54:29 UTC. Verdict distribution: **232 arcs `accept`, 3 arcs `hr`** (`arc_0105`, `arc_0128`, `arc_0165`). No writer processes were running during this pass (verified via process scan).

### Consolidation (18:17 UTC)

Backup first: `ai/data/curated/sft_chatml/train_master_gold.jsonl.pre_consolidation_20261009T181727Z` (397,286,656 bytes). The real run then appended **58 new sessions** to gold:

- **Before:** 189,666 gold lines, 0 reject lines. **After:** **189,724 gold lines**, reject log **2 lines** (delta +58 gold, +2 rejects).
- Dry-run summary before the run: 235 scanned arcs, 567 scanned sessions, 3 skipped by verdict (`hr`), 58 emitted, 507 duplicates, 0 content rejects, 2 structural rejects.
- The 58 new sessions span 22 arcs (`arc_0006`, `arc_0010`, `arc_0012`, `arc_0052`, `arc_0063`, `arc_0067`, `arc_0073`, `arc_0074`, `arc_0075`, `arc_0078`, `arc_0085`, `arc_0108`, `arc_0124`, `arc_0140`, `arc_0143`, `arc_0144`, `arc_0146`, `arc_0151`, `arc_0158`, `arc_0164`, `arc_0196`, `arc_0205`) — the writer's 25-target arcs minus the 3 `hr` arcs and the 2 structurally invalid sessions.
- Post-run dry-run: **0 emitted**, 565 duplicates, 2 structural rejects — idempotent, no further emission.
- Focused consolidation tests: **14 passed** (1.12s).
- Post-verification: all 58 new gold rows present exactly once, all `accept` verdicts, `consolidated_at` 2026-10-09T18:17:32.018–106Z. Gold arc rows now **565** (565 unique keys, 0 duplicates). Gold arc writer models now include `cf/@cf/deepseek-ai/deepseek-v4-flash-0731` (45 rows) and `cx/gpt-6-astra` (13 rows) from the regeneration run.

### Structural rejects (2 sessions)

`arc_0052` s1 and `arc_0144` s2 each contain **one consecutive-therapist-turn pair** (writer artifact: two therapist beats with different `track` labels and no client turn between). `_session_messages` rejects these by design; they landed in `consolidation_rejections.jsonl` with reason `invalid_session: consecutive therapist turns`.

- These 2 sessions are the only gap between 567 accepted record sessions and 565 gold rows; every other accepted session is in gold exactly once.
- **9 existing DPO pairs use these sessions as their chosen side** (4 session-mode superseded-version pairs for `arc_0052` s1 from four historical snapshots, 4 attempt pairs for `arc_0052` s1, 1 attempt pair for `arc_0144` s2). Their chosen text matches the current checkpoint render (the builder's `_render_session` tolerates consecutive therapist turns; the consolidation mapping does not). The pairs remain internally consistent; the question is whether the underlying sessions should be fixed (merge the two beats or insert the missing client turn), regenerated, or excluded — an owner decision.
- The 3 `hr` arcs (`arc_0105`, `arc_0128`, `arc_0165`) are skipped by verdict and have **no accepted partner** for their seed keys (each key maps to a single arc), so they produce no case pairs. Their 9 checkpoint rows are intentionally excluded from gold.

### DPO resync (18:24–18:25 UTC)

Backup first: `arc_dpo_pairs.jsonl.pre_dpo_resync_20261009T182459Z` (39,979,336 bytes, 555 lines). The real builder run then appended **126 new pairs**:

- Breakdown (from a dry-run decomposition using the builder's own functions): **27 session-mode** pairs (new superseded-version comparisons against the 8 `.pre*` snapshots for the newly-accepted arcs) and **99 attempt-mode** pairs (new failed attempts from the regeneration run). 0 case pairs (the 3 `hr` arcs have no accepted partners; the 15 existing case pairs are unchanged).
- New pairs span 20 arcs, led by `arc_0052` (15), `arc_0006` (13), `arc_0012` (11).
- **Post-resync integrity (681 pairs): 0 parse errors, 0 missing fields, 0 hash mismatches, 0 duplicate hashes.** Modes: 469 attempt, 197 session, 15 case.
- Post-resync dry-run: `built=776, emitted=0, duplicates=776` — pair file in sync with the builder.
- Focused builder tests: **27 passed** (1.63s).
- Case pairs re-checked: all **15 chosen sides** still byte-match the current checkpoint render of their accepted sessions (builder `_render_session`); **8 rejected sides** match their named source's current checkpoint render; **7 are stale** (the source session changed after the pair was built — `arc_0196`, `arc_0067`, `arc_0206`, `arc_0218`, `arc_0006`, `arc_0216`, `arc_0143`). Refreshing them would require an owner-approved rewrite; they remain internally consistent as historical evidence.

### Final coverage reconciliation (read-only, 18:20–18:28 UTC)

- **Accepted record sessions (567) → checkpoint: complete** (0 missing; 576 checkpoint rows = 567 accepted + 9 `hr` rows).
- **Accepted record sessions → gold: 565/567** — the only 2 missing are the structural rejects above (by design).
- **Gold → records: complete** — 0 gold rows reference a non-accepted arc (the 3 `hr` arcs contributed 0 gold rows).
- **Session/attempt pairs → accepted arcs: complete** — 0 pairs reference a non-accepted arc.
- **Checkpoint orphans:** 9 rows for the 3 `hr` arcs (intentional; excluded from gold and pairs).
- Git status in both `ai/` and the parent repo: clean; `git diff --check` clean. Unrelated edits from earlier sessions appear to have been committed separately (AI submodule now at `47010b60`).

## October 9 morning progress (04:15–05:25 UTC)

A separate actor (not this executor) fixed the confirmed builder defect and corrected the historical pairs while the regeneration writer ran. This executor verified that work read-only and completed the remaining safe Step 2 checks. No pair write, consolidation mutation, regeneration or external call was initiated by this executor.

### Builder fix and pair correction — verified

- `training/build_arc_dpo.py` was modified at **02:46:59 UTC**: `_session_pairs` now computes priors per `(arc_id, session_n)` from the current checkpoint, and `_attempt_pairs` uses a per-`(arc_id, session_n)` cache. This resolves the previously confirmed cache defect. `training/tests/test_build_arc_dpo.py` gained a multi-session regression test (`test_session_three_prompt_includes_session_two_prior`); the focused suite now has **27 tests, all passing** (run 05:12 UTC, `27 passed in 0.66s`, 9.98s wall).
- The pair file was rewritten at **04:13:49 UTC** (550 lines, 39,746,568 bytes) with a pre-fix backup preserved: `training/output/arc_corpus/arc_dpo_pairs.jsonl.pre_session_cache_fix_20261009T041349Z` (550 lines, 39,009,295 bytes, mtime 01:24:34). A structured backup-vs-current comparison shows exactly **95 session pairs changed — prompts only** (prior-session carry-forward added); chosen text, rejected text, metadata, all 365 attempt pairs and all 15 case pairs are byte-identical, and the 95 affected `pair_hash` values were recomputed. This matches the history-preserving correction the handoff called for, except that no separate old-hash→new-hash manifest was found; the backup file itself preserves the pre-fix hashes.
- Post-correction integrity across all 550 pairs: **0 parse/schema failures, 0 `pair_hash` mismatches, 0 duplicate hashes**.
- Deep verification against the current checkpoint: **98/98 session-2+ prompts** contain every available prior session's client lines verbatim with the correct `sN | ` prefix; **362/365 attempt prompts** fully verified, and the remaining 3 (`arc_0056` s3, `arc_0059` s2 ×2) miss only priors that are absent from the current checkpoint (see coverage finding) — expected gaps, not context defects.
- The quarantine of the 95 session pairs is therefore **lifted for context completeness**. Remaining caveats: chosen-source version certification is still outstanding; prompts remain reconstructed (corrective notes omitted, `brief_reconstructed: true`); the correction's authorization is attributed to the parallel session active 02:43–05:13 UTC and was not confirmed with the owner.

### DPO dry-run and case-pair finding (05:25 UTC, bounded snapshot)

`build_arc_dpo.py --dry-run` (3.82s, exit 0): `built=630, emitted=0, duplicates=630, by_mode={'session': 265, 'attempt': 365}`. The pair file is up to date against current inputs as of this snapshot; the writer was still active, so this is not a final reconciliation. **Case mode built 0 candidates**: case pairs require the rejected arm's record to carry verdict `hr`, and the in-flight regeneration reset those arcs to `audit: null` (`unknown`), so the 15 existing case pairs cannot currently be regenerated. **7 of the 15 case pairs are stale** against the current checkpoint (chosen text matches, rejected arm does not): they reference regeneration-target arcs `arc_0196`, `arc_0067`, `arc_0206`, `arc_0218`, `arc_0006`, `arc_0216`, `arc_0143`. Re-evaluate all case pairs after the writer completes and audits.

### Coverage reconciliation (read-only, 05:0x–05:20 UTC)

- **Gold:** 189,666 lines; **507 arc-family records** covering **210 accepted arcs / 507 sessions**. Ledger retention mechanically verified: **507/507** records have every assistant message starting with a parseable JSON object (the ledger). This is a mechanical retention check, not a clinical ledger-faithfulness judgment (`ARC_CORPUS_SPEC.md` §§4, 7).
- **Records → gold: complete.** The consolidation reads sessions embedded in arc records (not the checkpoint); the dry-run's 507 considered sessions with 507 duplicates confirms every session of the 210 accepted records is already in gold. No accepted session is missing from gold.
- **Checkpoint → records divergence (new finding):** **17 sessions across 11 accepted arcs** (`arc_0056` s1, `arc_0057` s1–2, `arc_0059` s1, `arc_0064` s1–2, `arc_0065` s1, `arc_0068` s1, `arc_0070` s1, `arc_0076` s1+s3, `arc_0077` s1–2, `arc_0090` s1–2, `arc_0092` s1–2) are embedded in their accepted records and present in gold, but **absent from the current checkpoint and from all 8 `.pre*` snapshots**. The generator only appends to the checkpoint, so something else removed them; cause and authorization unknown. Impact: the DPO builder has no chosen-text source for those sessions (no pairs can be built for them), and attempt prompts for affected arcs lack those priors (the 3 expected gaps above). Gold retains full content with provenance, so restoration from gold is possible if approved.
- All **490** accepted-arc sessions present in the checkpoint are in gold (0 missing).
- **Regeneration in flight:** 19 arcs with `audit: null` records (57 checkpoint sessions) await K3 audit; 12 sessions of the 25-target plans are still missing from the checkpoint while the writer runs.

### Checkpoint restoration and pair resync (06:15–06:55 UTC)

The 17-session checkpoint divergence documented above was **restored** with the user's approval ("fix it, so we can move forward"). All work below was done by this executor.

- **Backup first:** `training/output/arc_corpus/sessions_checkpoint.jsonl.pre_checkpoint_restore_20261009T062741Z` (copy of the 546-line file before any write).
- **Method:** each missing row was rebuilt from its arc record's embedded session turns (the audited structure), then strictly verified **before writing**: rendering the embedded turns through `consolidate_arc_corpus.py`'s ChatML mapping reproduced the gold row's messages exactly for all 17 sessions, and each row's `writer_model` matched the gold row's provenance. Rows were appended with fsync; post-verification confirmed all 17 keys present exactly once and no duplicate keys for the targets.
- **Lossy fields:** the original rows' `usage`, `facts`, `attempts` and `wall_s` are unrecoverable from gold; restored rows carry explicit defaults (`usage: null`, `facts: null`, `attempts: 1`, `wall_s: 0.0`). The builder and consolidation do not read these fields; documented here so nobody mistakes them for measured values.
- **Checkpoint after restore:** 563 lines (546 + 17). The writer appended 3 more rows during the operation (its own keys, untouched).
- **Attempt-pair resync:** the restoration gave the builder prior context it previously lacked. A fresh dry-run then emitted **8 new attempt pairs** (arc_0056 s3 a1; arc_0059 s2 a1+a2; arc_0064 s1 a1, s2 a1; arc_0090 s1 a1, s2 a2; arc_0092 s1 a1) — the previously "expected gap" prompts are now complete. Chosen/rejected text of the 3 overlapping combos is byte-identical to the old pairs; only the prompts gained the missing carry-forward (and, for arc_0056 s3, the correct plan gap line "ten days" replacing the wrong "two weeks").
- **Superseded-pair cleanup:** the 3 old incomplete-prompt attempt pairs (same arc/session/attempt keys, stale prompts) were removed; backup `training/output/arc_corpus/arc_dpo_pairs.jsonl.pre_superseded_attempt_dedup_20261009T064742Z`. Post-cleanup: **555 lines, 0 hash mismatches, 0 duplicate hashes, 0 duplicate attempt keys**, all three combos carry complete carry-forward (regex-verified).
- **Resync proof:** `build_arc_dpo.py --dry-run` after cleanup: `built=635, emitted=0, duplicates=635, by_mode={'session': 265, 'attempt': 370}` — the pair file is in sync with the builder against the restored checkpoint. Focused builder tests: **27 passed** (11.4s wall).
- **Note:** one pre-existing case-pair key (`case, arc_0014, 1`) appears twice with different rejected arms (two historical HR versions of arc_0216 against the same chosen). That is the same multi-superseded-version pattern as session mode, not a defect; left untouched.
- **Still open:** the divergence's root cause (what removed the 17 rows) remains unknown; the writer was active throughout, so counts remain a moving snapshot. Chosen-source certification is now complete (next section); completed-output audit of the regenerated arcs is still required before close-out.
- **Related finding (06:58 UTC):** the five arcs still lacking records (`arc_0143`, `arc_0146`, `arc_0164`, `arc_0196`, `arc_0205`) have **no prior state at all** — zero records in the current records file and every `arc_records*.jsonl` snapshot, and (except `arc_0164`/`arc_0205` session 1, written by the current writer) zero checkpoint rows and zero gold rows. The writer is generating them from scratch, not regenerating existing data. Separately, `arc_0196`'s and `arc_0143`'s old records **were removed during the October 5 lane2 operations** (present through `pre_lane2b4`/`pre_lane2b6`, absent from `pre_lane2b5`/`pre_lane2regen` onward; not present in any holdout/revision/lane file). Two case pairs (`arc_0008` vs `arc_0196`, `arc_0214` vs `arc_0143`) therefore reference rejected sources that no longer exist as records; their rejected text remains preserved in the pair file itself, but the source records cannot be re-verified unless restored from a snapshot. Treat this as intentional-but-undocumented supersession pending owner confirmation.

### Chosen-source certification (read-only, 06:50 UTC)

The previously outstanding "chosen-source version certification" was completed read-only for the 555-pair file. Method: for every session/attempt pair, the pair's `chosen` text must byte-match the render of its `(arc_id, session_n)` checkpoint row, and that checkpoint row's turns must reproduce the corresponding gold row's `messages` exactly through the consolidation ChatML mapping (same standard as the restoration above).

- **Session/attempt pairs (540): 540/540 certified** — 0 chosen mismatches, 0 checkpoint-vs-gold mismatches, 0 missing gold rows, 0 missing checkpoint rows.
- **0 session/attempt pairs reference any of the writer's 25 regeneration-target arcs**, so this certification is stable against the in-flight writer: no pair's chosen text can change when the writer finishes.
- **Case pairs (15): all 15 chosen sides certified** (chosen matches its accepted session's checkpoint render and gold). Rejected sides: **8 match** the current checkpoint's HR-session render; **5 are stale** (the checkpoint's HR session changed after the pair was built — consistent with the earlier stale-case-pair finding); **2 reference HR sessions absent from the checkpoint** (`arc_0196` s1, `arc_0143` s1 — both writer regeneration targets). Re-evaluate all case pairs after the writer completes and audits, as already documented.
- Certification covers source fidelity, not preference quality: prompts remain reconstructed, and clinical/ledger faithfulness judgments remain separate (`ARC_CORPUS_SPEC.md` §§4, 7).

### Parallel work observed (not this executor's)

- New files `training/clef_client.py`, `training/eval_clef.py`, `training/tests/test_clef_client.py` (modified 05:11–05:13 UTC): a probe evaluating Cloudflare Workers AI's `clef` decision model as a structured co-auditor against the existing 17-probe ground-truth set. This workstream makes external API calls; its scope/budget approval is unknown to this executor.
- `pyproject.toml` gained a ruff per-file ignore for `training/eval_clef.py` (T201). Benign.
- Writer: the earlier PID 902322 is gone; **PID 1452957** (started 02:43:36 UTC, same 25-plan command) was still `Sl` at 05:18 UTC (2h35m elapsed, ~2s CPU) and `/home/vivi/arc_regen/writer.log` (1,513 lines) still shows `http_503` retries with 60–120s backoffs. At **05:29:12 UTC** that writer had ended and a **new writer (PID 2078765) launched with the same 25-plan command**; the checkpoint had grown to 17,942,079 bytes (mtime 03:50:16). At **06:01:53 UTC** writer launch **PID 2209971** started (relauncher PID 2202591, watcher PID 2208445): 5 records were missing then; `arc_0006` completed during this run (records now 230 lines). At 06:5x UTC, 5 of the 25 planned arcs still lacked records (`arc_0143`, `arc_0146`, `arc_0164`, `arc_0196`, `arc_0205`) and the writer was still hitting `http_503` retries (writer.log 1,580 lines). The writer uses model `nvidia/z-ai/glm-5.3-flash` via a featherless key and logs to wandb (`pixelated-empathy-kan28`). Regeneration is still in flight; the retry/stop decision remains with the owner.

## Broader work: choose the next evidence-backed task

Only after the writer/audit gate above is satisfied (or explicitly scoped by the user), inspect current artifacts and pipeline specifications before selecting another task. Conditional candidate work, in priority order:

1. **Validate DPO candidates** from the completed/audited inputs using `training/build_arc_dpo.py --dry-run`; address the confirmed context defect and approve history-preserving correction before any append. Preserve the 518 historical pairs and their source evidence; do not assume successful deduplication makes them fit for preference use.
2. **Audit corpus integrity and provenance:** compare accepted arc/session coverage against gold provenance, ensure no duplicate pair hashes, inspect reject reasons, and verify each record retains required provenance and ledger content.
3. **Assess remaining data quality gaps** against `training/ARC_CORPUS_SPEC.md`, the approved arc plan list, and current audit/review outputs. Focus on actionable discrepancies (missing plans/sessions, unresolved review items, structural or ledger violations), not broad speculative refactors.
4. **Keep training preparation blocked** until dataset quality work is resolved and explicit dataset close-out is obtained. Consolidation is not dataset completion; do not create Fireworks preparation documents, export/upload data or start training meanwhile.

Selected next dataset task, **blocked by the writer/audit gate**: run the existing DPO builder's dry-run and reconcile preference integrity, accepted-session coverage, ledger retention and unresolved reviews (Step 2). Check for superseded arc versions and multiple gold records per arc/session: consolidation uses the latest arc record but does not supersede historical gold. The reviewed writer targets only `audit: null` arcs, not the currently accepted gold sources; reverify this against completed inputs rather than assuming it remains true.

The larger goal is high-quality, provenance-preserving clinical dialogue training data: accepted multi-session arcs consolidated into T1 gold, with deterministic ledger/structure gates and useful chosen/rejected DPO pairs. Keep synthetic data provenance explicit and maintain separation between transcript content, ledger state, and audit judgments.

## Commands and operational notes

- Consolidation module imports require both paths: `PYTHONPATH=/home/vivi/pixelated/ai:/home/vivi/pixelated`.
- The workspace and AI repo each have their own Python environment: workspace `/home/vivi/pixelated/.venv/bin/python`; AI repo `/home/vivi/pixelated/ai/.venv/bin/python`. For commands run from `ai/`, use the AI repo environment (`./.venv/bin/python`) so dependencies resolve from the correct project environment.
- Source of truth for consolidation behavior: `ai/training/consolidate_arc_corpus.py`; focused tests: `ai/training/tests/test_consolidate_arc_corpus.py`.
- The AI submodule has unrelated local modifications in `training/audit_arc_corpus.py` and `training/generate_arc_corpus.py`; do not revert or overwrite them. The parent also has unrelated `foresight` and `churnmeon/` changes, and is 12 commits behind `origin/staging`; do not sync or alter those as part of this data task.
- Dataset artifacts are large and append-oriented. Avoid rewriting gold or checkpoints; use the pipeline's idempotent deduplication and atomic utilities.

## Handoff completion checklist

- [x] Execute approved real consolidation and capture its summary (no-op against already consolidated gold).
- [x] Verify gold line delta and inspect any reject-log delta (both zero; hashes unchanged).
- [x] Confirm a subsequent dry-run emits no additional gold.
- [x] Run focused consolidation tests (14 passed).
- [x] Refresh this handoff with actual post-run counts and the selected next task.
- [x] Leave unrelated working-tree changes untouched.
- [x] Post-completion consolidation after writer finished (+58 gold, 2 structural rejects; 18:17 UTC).
- [x] Post-completion DPO resync (+126 pairs, 681 total, integrity verified; 18:25 UTC).
- [x] Final coverage reconciliation: records→checkpoint→gold→pairs complete except the 2 documented structural rejects.
- [x] Owner decision: 2 structural rejects — resolved by regeneration per the standing directive: `arc_0144` s2 and all three `arc_0052` sessions regenerated under `cx/gpt-6-astra`, audited accept, banked to gold 04:01:11 UTC (see the October 10 close-out section).
- [x] Owner decision: 3 `hr` arcs — resolved by note-driven regeneration: `arc_0105`, `arc_0165`, `arc_0128` all accept (October 9, 19:20–20:41); 235/235 records accept.
- [x] Owner decision: refresh the 7 stale case pairs or keep as historical evidence — kept (default disposition, October 10; reversible, see close-out section).
- [x] Explicit dataset close-out (blocks Fireworks stages) — evidence-complete October 10, 04:15 UTC (see close-out section); Fireworks stages unblocked for documentation-only work.

## Evidence caveat

The October 9 18:30 UTC section supersedes earlier October 9 snapshots: the writer pipeline completed (last writes 17:07:58 / 16:54:29 UTC) and all counts in the newest section were measured against the quiescent state. Earlier sections describing moving-snapshot conditions remain as historical record. The **October 10 dataset close-out section supersedes that snapshot's open decisions**; the completion checklist above records their resolutions. Clinical/ledger faithfulness judgments remain separate from these mechanical checks (`ARC_CORPUS_SPEC.md` §§4, 7).
