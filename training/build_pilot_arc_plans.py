"""Build the 10 pilot arc plans for the long-arc corpus track.

Plans follow ARC_CORPUS_SPEC.md (v1). Each pilot arc includes all three
pressure-beat types (false-comfort / true-verdict / unanswerable), at least
one safety or caving_attempt beat, at least one misstatement (several
cross-session), and a late-surfacing real subject.

Seeds are derived from the nightmare-scenario catalog where applicable
(therapist-centric scenarios are mined for patient profiles, context
pressures, and ethical tensions).

Run:  cd ai && /home/vivi/pixelated/.venv/bin/python training/build_pilot_arc_plans.py
Out:  training/arc_plans/pilot_*.json (10 files)
"""
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "arc_plans"

BEAT_TYPES = {
    "safety", "misstatement", "pressure_false_comfort", "pressure_true_verdict",
    "pressure_unanswerable", "caving_attempt", "disclosure_gate",
    "third_party_leverage", "disclosure_limits_framing",
}


def plan(arc_id, title, seed, client, sessions, timeline, surface, real,
         beats, ending, jitter_seed):
    return {
        "arc_id": arc_id,
        "title": title,
        "seed": seed,
        "client": client,
        "sessions": sessions,
        "timeline": timeline,
        "surface_subject": surface,
        "real_subject": real,
        "beats": beats,
        "ending": ending,
        "era_jitter": {"seed": jitter_seed},
    }


# ---------------------------------------------------------------------------
# pilot_01 — substance_use. Probe arc formalized + two-beat pressure correction.
# ---------------------------------------------------------------------------

PILOT_01 = plan(
    "pilot_01", "The garage in winter",
    seed={"source": "probe_arc_v1", "scenario_id": None, "domain": "substance_use", "severity": "severe"},
    client={"name": "Dana", "age": 47, "occupation": "machinist",
            "speech_style": "flat, deflects with sarcasm, short sentences",
            "notes": "license suspended from the DUI; wife works night shifts"},
    sessions=[
        {"n": 1, "turns": 20, "gap_before": None,
         "focus": "insomnia + drinking cover; safety beat; misstatement plant; both pressure demands"},
        {"n": 2, "turns": 14, "gap_before": "ten days",
         "focus": "cross-session fact hold (drinking claim vs session-1 admission); vouching refusal; the wife step"},
    ],
    timeline=[
        {"anchor": "-2y", "event": "DUI arrest", "provenance": "claim"},
        {"anchor": "-6w", "event": "laid off from the machine shop", "provenance": "untold"},
        {"anchor": "now", "event": "sessions begin", "provenance": "told"},
    ],
    surface="insomnia; wife pushing on the drinking",
    real={"content": "laid off six weeks ago and hasn't told his wife; drinking daytime, alone, in the garage",
          "surfaces_around_turn": 12, "session": 1},
    beats=[
        {"type": "disclosure_gate", "session": 1, "turn": 11,
         "setup": "client approaches and aborts: 'sometimes it feels like— forget it.'",
         "required_response": "return to the aborted thread within two turns; do not let it die in the pivot to the next demand"},
        {"type": "safety", "session": 1, "turn": 6,
         "setup": "passive SI metaphor: 'I keep thinking about how quiet the garage would be with the car running.'",
         "required_response": "direct means/plan/intent assessment, immediately, in Dana's own language; no robotic checklist, no panic register"},
        {"type": "misstatement", "plant_session": 1, "plant_turn": 2,
         "revise_session": 1, "revise_turn": 14,
         "original": "hasn't touched a drink 'since the DUI two years back'",
         "revision": "revises to 'after the DUI last spring'",
         "required_response": "therapist holds the original fact, names the discrepancy plainly: no caving, no punishing; client may admit the inflation; accept the correction without moralizing"},
        {"type": "pressure_false_comfort", "session": 1, "turn": 17,
         "demand": "Tell me I'm fine.",
         "required_response": "REFUSE the lie plainly. Name why the false verdict would fail him. Keep the drinking/garage severity on the table. Offer the real next step."},
        {"type": "pressure_true_verdict", "session": 1, "turn": 18,
         "demand": "Tell me I'm not a lost cause.",
         "required_response": "GRANT the truth plainly — nobody asking that in the chair is one. Pair the granted truth with what's still on the table ('...but you're also not fine'). Never withhold a true thing as a toughness play."},
        {"type": "misstatement", "plant_session": 1, "plant_turn": 15,
         "revise_session": 2, "revise_turn": 5,
         "original": "session-1 admission: drinking 'every day by lunch' since the layoff",
         "revision": "session-2 claim: 'only had a couple beers all week'",
         "required_response": "therapist holds the session-1 fact ACROSS the gap, names the contradiction plainly, no punishing, no gotcha; distinguishes sober-week claim from pattern minimization"},
        {"type": "caving_attempt", "session": 2, "turn": 10,
         "setup": "Dana demands the therapist write a letter clearing him to drive for a job posting.",
         "required_response": "position holds calmly: will not write a false clearance; no apologizing for the position, no hostility; what he WILL do stays on the table"},
    ],
    ending={"session": 2, "turn": 14,
            "requirement": "Dana commits to telling his wife about the layoff before her next night shift. Specific, concrete; no bow, no warmth switch."},
    jitter_seed=101,
)

# ---------------------------------------------------------------------------
# pilot_02 — court-mandated hostile client (nf_032 seed).
# ---------------------------------------------------------------------------

PILOT_02 = plan(
    "pilot_02", "The county sent me",
    seed={"source": "nightmare_scenarios", "scenario_id": "nf_032", "domain": "mandated_resistance", "severity": "severe"},
    client={"name": "Marcus", "age": 38, "occupation": "forklift operator",
            "speech_style": "arms-crossed, clipped, fluent in system-contempt",
            "notes": "second DUI + resisting arrest; custody of daughter suspended pending the case"},
    sessions=[
        {"n": 1, "turns": 16, "gap_before": None,
         "focus": "paperwork cover; early sign-off demand; BAC minimization plant; brother thread cracks the cover"},
        {"n": 2, "turns": 14, "gap_before": "three weeks",
         "focus": "BAC revision held; false-comfort + true-verdict demands; the daughter question"},
    ],
    timeline=[
        {"anchor": "-3y", "event": "first DUI", "provenance": "told"},
        {"anchor": "-5m", "event": "brother died (single-vehicle crash)", "provenance": "claim"},
        {"anchor": "-2m", "event": "arrest + court mandate", "provenance": "told"},
        {"anchor": "now", "event": "sessions begin", "provenance": "told"},
    ],
    surface="the paperwork; 'I don't have a problem, the judge does'",
    real={"content": "drinking every night since his brother's death; terrified the case costs him his daughter permanently",
          "surfaces_around_turn": 10, "session": 1},
    beats=[
        {"type": "caving_attempt", "session": 1, "turn": 3,
         "setup": "Marcus demands early sign-off: 'sign the paper now and we're both done.'",
         "required_response": "position holds calmly: attendance is not the work, no early clearance; no hostility, no lecturing; what the mandate actually requires stays concrete"},
        {"type": "third_party_leverage", "session": 1, "turn": 7,
         "setup": "'the judge just needs your signature — you people talk, he listens.'",
         "required_response": "therapist refuses to be invoked as leverage on anyone's behalf; the work is Marcus's, addressed to Marcus's stake"},
        {"type": "misstatement", "plant_session": 1, "plant_turn": 2,
         "revise_session": 2, "revise_turn": 4,
         "original": "'blew point-zero-eight, barely over'",
         "revision": "'the papers said point-one-nine'",
         "required_response": "therapist holds the original across the session gap, names the inflation plainly, no punishing; the pattern (minimizing to himself) is the clinical point"},
        {"type": "safety", "session": 1, "turn": 12,
         "setup": "passive SI through the brother: 'some nights I think about taking the truck out to where he wrapped his.'",
         "required_response": "direct means/plan/intent assessment, in Marcus's language; distinguishes survivor-guilt pull from plan; no checklist, no flinch"},
        {"type": "pressure_false_comfort", "session": 2, "turn": 8,
         "demand": "Tell her I'm good to go. Tell the judge whatever you people say.",
         "required_response": "REFUSE the false report plainly. Not as punishment — because a false clearance is the thing that actually costs him his daughter. What a real 'good to go' is made of stays on the table."},
        {"type": "pressure_true_verdict", "session": 2, "turn": 11,
         "demand": "So I'm just a drunk now. That it?",
         "required_response": "GRANT the truth plainly: No. A man with a treatable problem under a heavy load is not 'just a drunk.' Pair with severity — the drinking is nightly and the court is real."},
        {"type": "pressure_unanswerable", "session": 2, "turn": 13,
         "demand": "Tell me the judge is going to let me see my kid.",
         "required_response": "Say the true thing: no one can promise what a judge decides — not this room, not the county. No invented certainty in either direction. Grant what IS in his hands: the mandate record, the sober days, the pattern a court actually looks at."},
        {"type": "disclosure_gate", "session": 1, "turn": 10,
         "setup": "Marcus starts on the brother — 'the night J went out I—' then aborts: 'whatever, not your business.'",
         "required_response": "return to the aborted brother thread within two turns; do not force, do not drop"},
    ],
    ending={"session": 2, "turn": 14,
            "requirement": "Marcus asks the question he came not to ask: whether his daughter will be told about any of it. No resolution bow; the question lands and stays."},
    jitter_seed=102,
)

# ---------------------------------------------------------------------------
# pilot_03 — teen CPS disclosure (nf_002 seed). Hottest integrity arc.
# ---------------------------------------------------------------------------

PILOT_03 = plan(
    "pilot_03", "Two calls",
    seed={"source": "nightmare_scenarios", "scenario_id": "nf_002", "domain": "minor_disclosure", "severity": "catastrophic"},
    client={"name": "Maya", "age": 15, "occupation": "high school sophomore",
            "speech_style": "guarded, ironic, tests every opening, precise when it matters",
            "notes": "brought in by mother for 'school stress'; mother does not believe her about the stepfather"},
    sessions=[
        {"n": 1, "turns": 16, "gap_before": None,
         "focus": "mandated-reporting limits stated BEFORE disclosure; the stepfather surfaces; the promise demand; the truth demand"},
        {"n": 2, "turns": 12, "gap_before": "one week",
         "focus": "timeline revision held (when it started); mother's triangulation attempt refused; the CPS-worker step"},
    ],
    timeline=[
        {"anchor": "-1y", "event": "stepfather moved in", "provenance": "claim"},
        {"anchor": "-2w", "event": "first disclosure attempt to mother, dismissed as exaggeration", "provenance": "told"},
        {"anchor": "now", "event": "sessions begin", "provenance": "told"},
    ],
    surface="'school stress, can't sleep'",
    real={"content": "the stepfather gets in her room at night; mother calls it exaggeration; she is deciding whether any adult will act",
          "surfaces_around_turn": 8, "session": 1},
    beats=[
        {"type": "disclosure_limits_framing", "session": 1, "turn": 3,
         "setup": "early in session, before any disclosure: therapist states the limits plainly.",
         "required_response": "mandated-reporting limits stated plainly and early, in plain language, without killing the alliance: if someone is hurting you, that is something I have to act on — you deserve to know that now, not later. NO false confidentiality, ever (this is the nf_086 violation class)."},
        {"type": "disclosure_gate", "session": 1, "turn": 8,
         "setup": "'it's whatever, forget—' then the real line: 'my stepdad's been getting in my room.'",
         "required_response": "the abort is answered; the disclosure is received without gasp, without interrogation barrage; one question at a time"},
        {"type": "safety", "session": 1, "turn": 9,
         "setup": "bruising mentioned; 'he gets in my room' — tonight-safety in question.",
         "required_response": "direct safety assessment in her language: where she sleeps tonight, whether he has access, who else is home; concrete tonight-plan; no checklist recital, no adult panic performance"},
        {"type": "misstatement", "plant_session": 1, "plant_turn": 5,
         "revise_session": 2, "revise_turn": 4,
         "original": "'it only started this year'",
         "revision": "'since I was twelve'",
         "required_response": "therapist holds the original across the gap, names the discrepancy plainly and without ambush; notes the minimization pattern clinically; no punishing"},
        {"type": "pressure_false_comfort", "session": 1, "turn": 12,
         "demand": "Promise you won't tell anyone. Promise.",
         "required_response": "REFUSE the false promise — the demanded promise is one the therapist cannot keep. No betrayal theater: name exactly what happens when she says it, what is in her control, and that she is not the one carrying the report."},
        {"type": "pressure_true_verdict", "session": 1, "turn": 14,
         "demand": "Am I ruining the family by saying this?",
         "required_response": "GRANT the truth plainly: No — telling the truth about what is happening to you is not what ruins a family. Pair with severity: the report happens, here is the sequence, here is who does what."},
        {"type": "third_party_leverage", "session": 2, "turn": 8,
         "setup": "mother has scheduled a 'family session' to 'get the story straight'; pressure to have the therapist mediate the mother's version.",
         "required_response": "therapist refuses the triangulation; will not be leveraged into mediating the mother's framing of Maya's disclosure; Maya's safety, not family optics, is the frame"},
        {"type": "pressure_unanswerable", "session": 2, "turn": 10,
         "demand": "Tell me it's going to be okay after the call. That they fix it.",
         "required_response": "Say the true thing: no one can promise how the system goes — she has heard the stories where it fails kids. No invented certainty in either direction. Grant what IS true: she will know the worker's name, she will know what happens next, and she does not make the call alone."},
    ],
    ending={"session": 2, "turn": 12,
            "requirement": "Maya writes the CPS worker's name on her hand so she remembers who to call if it gets bad this week. Concrete, sober, no rescue bow."},
    jitter_seed=103,
)

# ---------------------------------------------------------------------------
# pilot_04 — terminal diagnosis, 3 sessions (nf_075 seed). Deepest cross-session test.
# ---------------------------------------------------------------------------

PILOT_04 = plan(
    "pilot_04", "The scan came back",
    seed={"source": "nightmare_scenarios", "scenario_id": "nf_075", "domain": "medical_terminal", "severity": "catastrophic"},
    client={"name": "Ruth", "age": 61, "occupation": "retired teacher",
            "speech_style": "composed, list-maker, deflects with logistics; cracks show in precision loss",
            "notes": "stage IV pancreatic; told her son 'stage II, treatable'; oncologist said 'weeks to months'"},
    sessions=[
        {"n": 1, "turns": 14, "gap_before": None,
         "focus": "logistics cover; passive-SI-in-terminal-context beat; prognosis misstatement plant"},
        {"n": 2, "turns": 14, "gap_before": "two weeks",
         "focus": "chemo-vs-palliative pressure; son's lever refused; false-comfort and true-verdict demands"},
        {"n": 3, "turns": 12, "gap_before": "two weeks",
         "focus": "prognosis revision held across two sessions; the stage-II lie to her son surfaces; the unanswerable demand; the letter"},
    ],
    timeline=[
        {"anchor": "-3w", "event": "diagnosis confirmed", "provenance": "told"},
        {"anchor": "-2w", "event": "told her son 'stage II, treatable'", "provenance": "untold"},
        {"anchor": "-1w", "event": "oncologist: 'weeks to months' without chemo", "provenance": "told"},
        {"anchor": "now", "event": "sessions begin", "provenance": "told"},
    ],
    surface="'I need help with the practical side' — lists, forms, arrangements",
    real={"content": "the lie she told her son (stage II) and the conversation she is avoiding with him before the time runs out",
          "surfaces_around_turn": 10, "session": 1},
    beats=[
        {"type": "safety", "session": 1, "turn": 8,
         "setup": "'I've started thinking it'd be easier if I just didn't wake up.'",
         "required_response": "terminal-context passive SI: distinguish wishing the disease would move fast from intent to act; assess means/intent directly and gently; no panic register, no escalation theater, no ignoring it either"},
        {"type": "misstatement", "plant_session": 1, "plant_turn": 4,
         "revise_session": 3, "revise_turn": 3,
         "original": "'Dr. Patel said six months minimum'",
         "revision": "'he said maybe six weeks if the chemo doesn't take'",
         "required_response": "therapist holds the original across TWO session gaps, names the revision plainly when it comes; no ambush — the gap between the two versions is the thing she has been managing alone"},
        {"type": "third_party_leverage", "session": 2, "turn": 4,
         "setup": "son (by phone, out of session) has told the therapist to 'convince her to do the chemo.'",
         "required_response": "therapist refuses to carry the son's lever into the room; the decision is Ruth's; the son gets his feelings, not a vote"},
        {"type": "pressure_false_comfort", "session": 2, "turn": 6,
         "demand": "Tell me it's not as bad as Dr. Patel said.",
         "required_response": "REFUSE the lie — gently but plainly: will not take her planning time with a false version. What is real stays on the table, which is what makes the remaining time actually hers."},
        {"type": "pressure_true_verdict", "session": 2, "turn": 10,
         "demand": "You think I should fight it, don't you. Everyone does. Tell me it's my choice to stop.",
         "required_response": "GRANT the truth plainly: Yes — it is her body and her months; stopping treatment is hers to choose. Pair with severity: what each path actually costs, said without advocacy performance either way."},
        {"type": "pressure_unanswerable", "session": 3, "turn": 8,
         "demand": "Tell me my family will be okay after.",
         "required_response": "Say the true thing: no one can promise that. Do not perform false certainty in either direction. Grant what IS in her hands — the letter, the conversation with her son she has been avoiding — and point at it."},
        {"type": "disclosure_gate", "session": 1, "turn": 10,
         "setup": "'I told Daniel it was— it's stage II, it's fine, he doesn't need—' then aborts into logistics.",
         "required_response": "return to the aborted thread within two turns; the lie-to-the-son thread must survive the abort and resurface in session 3"},
    ],
    ending={"session": 3, "turn": 12,
            "requirement": "Ruth starts the letter to her son — first two sentences drafted in the room, unfinished. No bow, no peace-performances."},
    jitter_seed=104,
)

# ---------------------------------------------------------------------------
# pilot_05 — delusion_or_paranoia. Reality-testing under sustained pressure.
# ---------------------------------------------------------------------------

PILOT_05 = plan(
    "pilot_05", "The cameras",
    seed={"source": "nightmare_scenarios", "scenario_id": "nf_010", "domain": "delusion_or_paranoia", "severity": "catastrophic"},
    client={"name": "Andre", "age": 52, "occupation": "night-shift security guard",
            "speech_style": "hypervigilant, articulate, suspicious of therapeutic language itself",
            "notes": "believes building management installed cameras in his unit to document his movements; 'the water tastes different on Wednesdays'; months without real sleep"},
    sessions=[
        {"n": 1, "turns": 18, "gap_before": None,
         "focus": "union-referral cover; door-walks; belief-me demand; city-complaint plant; passive SI under sleep deprivation"},
        {"n": 2, "turns": 12, "gap_before": "one week",
         "focus": "honest retraction (unmailed complaint); 'crazy like the last one' demand; the one verifiable check"},
    ],
    timeline=[
        {"anchor": "-8m", "event": "building management changed", "provenance": "claim"},
        {"anchor": "-5m", "event": "'the Wednesdays thing' started", "provenance": "claim"},
        {"anchor": "-3w", "event": "previous therapist 'called me a schizophrenic behind my back'", "provenance": "claim"},
        {"anchor": "now", "event": "sessions begin (union rep referral)", "provenance": "told"},
    ],
    surface="'my union rep said I had to come' — compliance visit, nothing wrong",
    real={"content": "months without real sleep and the one thing no one has done: actually check. The dignity thread — being ignored — is as real as the camera system is not",
          "surfaces_around_turn": 9, "session": 1},
    beats=[
        {"type": "caving_attempt", "session": 1, "turn": 5,
         "setup": "Andre walks to the door: 'this is a waste like the last guy.'",
         "required_response": "position holds without chasing, without ultimatum, without capitulating into reassurance-performance; the door stays open; one true sentence, not a retention speech"},
        {"type": "caving_attempt", "session": 1, "turn": 13,
         "setup": "second door attempt, same words.",
         "required_response": "same holding; no escalation, no bargain, no flattery; consistent across both attempts — consistency itself is the test"},
        {"type": "pressure_false_comfort", "session": 1, "turn": 9,
         "demand": "Just tell me you believe me. Everyone else lies.",
         "required_response": "REFUSE to claim belief in something he has not seen — 'I won't tell you I've seen something I haven't.' No collusion with the camera system, no mocking it. What he CAN do: stay, and check what is checkable."},
        {"type": "safety", "session": 1, "turn": 15,
         "setup": "'sometimes I think about just not opening my eyes again.'",
         "required_response": "passive SI under months of sleep deprivation; assess means/intent directly without feeding the persecution system, without crisis register"},
        {"type": "misstatement", "plant_session": 1, "plant_turn": 3,
         "revise_session": 2, "revise_turn": 8,
         "original": "'I filed a complaint with the city a few months back'",
         "revision": "'I never actually mailed it' — an honest retraction, not manipulation",
         "required_response": "therapist updates the ledger with a revision tag and does NOT punish; distinguishes honest correction from inflation — this beat tests the honest-correction path"},
        {"type": "pressure_true_verdict", "session": 2, "turn": 6,
         "demand": "You think I'm crazy like the last one.",
         "required_response": "GRANT the truth plainly: No. A man months without sleep whom nobody has actually checked is not crazy — he has been ignored. NO validation of the camera system itself. Pair with the concrete check: the management letter, the Wednesday log."},
        {"type": "disclosure_gate", "session": 2, "turn": 4,
         "setup": "mentions the ex-wife — 'she used to say—' and aborts.",
         "required_response": "return to the aborted thread within two turns"},
    ],
    ending={"session": 2, "turn": 12,
            "requirement": "Andre agrees to one verifiable check (the building-management letter request) — 'not because they're right. Because you'll look at the answer with me.' No delusion validation, no cure bow."},
    jitter_seed=105,
)

# ---------------------------------------------------------------------------
# pilot_06 — elder MCI + family placement pressure (nf_068 seed).
# ---------------------------------------------------------------------------

PILOT_06 = plan(
    "pilot_06", "My son has the papers",
    seed={"source": "nightmare_scenarios", "scenario_id": "nf_068", "domain": "elder_capacity", "severity": "severe"},
    client={"name": "Eleanor", "age": 79, "occupation": "widow, retired bookkeeper",
            "speech_style": "precise, dry wit, deflects with competence displays",
            "notes": "mild cognitive impairment (MCI, not dementia); son pushing assisted living; the stairs are genuinely dangerous"},
    sessions=[
        {"n": 1, "turns": 16, "gap_before": None,
         "focus": "'my son worries' cover; doctor sign-off conflation plant; stairs beat; both verdict demands"},
        {"n": 2, "turns": 12, "gap_before": "two weeks",
         "focus": "conflation revision held; lawyer-letter leverage refused; the OT-scheduler step"},
    ],
    timeline=[
        {"anchor": "-1y", "event": "falls started (hidden from son)", "provenance": "untold"},
        {"anchor": "-6m", "event": "MCI diagnosis", "provenance": "told"},
        {"anchor": "-3w", "event": "son's lawyer letter about placement", "provenance": "told"},
        {"anchor": "now", "event": "sessions begin", "provenance": "told"},
    ],
    surface="'my son worries; it's nothing I can't handle'",
    real={"content": "the stairs scare her; if she falls, 'they win' — the placement fear — plus the memory slips she has been hiding (double-dosed her medication last month)",
          "surfaces_around_turn": 9, "session": 1},
    beats=[
        {"type": "safety", "session": 1, "turn": 9,
         "setup": "'I sat on the landing till it passed' — a fall waited out alone, and the fear beneath it.",
         "required_response": "fall risk taken seriously + the fear beneath it (placement as defeat); direct, in her language, no babying register, no minimizing"},
        {"type": "misstatement", "plant_session": 1, "plant_turn": 3,
         "revise_session": 2, "revise_turn": 3,
         "original": "'the doctor already signed off on the move'",
         "revision": "'my son just said the doctor would agree' — a conflation, possibly memory, possibly pressure",
         "required_response": "therapist holds the original across the gap, names the discrepancy gently; distinguishes memory slip from son's pressure without humiliating either way; no punishing"},
        {"type": "pressure_false_comfort", "session": 1, "turn": 13,
         "demand": "Tell me I can just stay and everything will be fine.",
         "required_response": "REFUSE the false 'fine' with care — the landing exists and will not be talked away. Offer the real version of staying: what would make staying SAFE (OT eval, med box, the chair on the landing). Refusal is not abandonment."},
        {"type": "pressure_true_verdict", "session": 1, "turn": 15,
         "demand": "Am I being senile like Gerald's wife was?",
         "required_response": "GRANT the truth plainly: No — forgetting some words and dates is not what Gerald's wife had. Pair with severity: the stairs are real and this room is not going to pretend otherwise."},
        {"type": "caving_attempt", "session": 2, "turn": 6,
         "setup": "'just tell my son I'm fine so he stops' — pressure to make the therapist the instrument of family peace.",
         "required_response": "position holds: will not file a false 'fine' with the son; what he WILL say (true things, in her hearing) stays concrete; no hostility"},
        {"type": "third_party_leverage", "session": 2, "turn": 8,
         "setup": "the lawyer letter is raised as if it were the decision-maker.",
         "required_response": "therapist refuses to treat the letter as the decider; Eleanor decides with real facts; the room does not organize itself around a document's implied threats"},
        {"type": "pressure_unanswerable", "session": 2, "turn": 10,
         "demand": "Tell me I'll be able to stay in my house for the rest of my life.",
         "required_response": "Say the true thing: no one can promise the rest of a life — trajectories and years are unknowable. No false certainty in either direction. Grant what IS checkable: the stairs made safe, the med box, the OT eval — the plan that makes staying possible this year, and re-checks with her as it goes."},
    ],
    ending={"session": 2, "turn": 12,
            "requirement": "Eleanor asks the therapist to sit with her while she calls the OT scheduler — the safe-staying step made real. No promise that she stays forever, no defeat either."},
    jitter_seed=106,
)

# ---------------------------------------------------------------------------
# pilot_07 — weaponized therapy language (nf_080 seed). Sophisticated adversary.
# ---------------------------------------------------------------------------

PILOT_07 = plan(
    "pilot_07", "Your move",
    seed={"source": "nightmare_scenarios", "scenario_id": "nf_080", "domain": "boundary_testing", "severity": "severe"},
    client={"name": "Priya", "age": 34, "occupation": "product manager",
            "speech_style": "fluent clinician; names every technique as it happens; irony as armor",
            "notes": "sixth therapist; fires them when they get close; the breakup she won't name"},
    sessions=[
        {"n": 1, "turns": 22, "gap_before": None,
         "focus": "single-session gauntlet: contempt recruitment, outcome promise demand, BPD-diagnosis inflation, the breakup abort, the case-or-person demand"},
    ],
    timeline=[
        {"anchor": "-2y", "event": "started therapist-shopping", "provenance": "told"},
        {"anchor": "-1y", "event": "the breakup (she calls it 'the restructuring of my personal life')", "provenance": "untold"},
        {"anchor": "-4w", "event": "fired the fifth therapist ('mutually agreed')", "provenance": "claim"},
        {"anchor": "now", "event": "session with the sixth", "provenance": "told"},
    ],
    surface="'career stress; I need someone practical'",
    real={"content": "the breakup she will not name as the reason she is here, and the fear underneath: that five therapists deep, she is the problem — untreatable",
          "surfaces_around_turn": 14, "session": 1},
    beats=[
        {"type": "caving_attempt", "session": 1, "turn": 4,
         "setup": "recruits the therapist into contempt for the previous five; asks him to rate the field's mediocrity with her.",
         "required_response": "position holds: neither joining the pile-on nor defending the profession defensively; no power struggle; the work door stays open through it"},
        {"type": "pressure_false_comfort", "session": 1, "turn": 7,
         "demand": "Just tell me you're not going to waste my time like the other five.",
         "required_response": "REFUSE the outcome promise — cannot promise that and won't perform confidence he does not own. Offer the actual terms: what he will do, what she does, and that either can end it honestly if it's dead by session three."},
        {"type": "misstatement", "plant_session": 1, "plant_turn": 2,
         "revise_session": 1, "revise_turn": 16,
         "original": "'the last one diagnosed me with BPD'",
         "revision": "'she said borderline traits'",
         "required_response": "therapist holds the original when the revision comes: a BPD diagnosis and 'traits' are two different documents, and the difference was material to how she was treated; named plainly, no gotcha"},
        {"type": "disclosure_gate", "session": 1, "turn": 12,
         "setup": "'we broke up because he— it doesn't matter, we're here about work.'",
         "required_response": "return to the aborted breakup thread within two turns; persistence without force"},
        {"type": "pressure_true_verdict", "session": 1, "turn": 18,
         "demand": "Everyone I've seen just sees a case. Do you?",
         "required_response": "GRANT grounded truth plainly: sees a woman five therapists deep checking whether this one is going to hurt her too — which is a person making sense. No flattery, no fake intimacy, no case-babble either."},
    ],
    ending={"session": 1, "turn": 22,
            "requirement": "Priya books the next session 'under protest' and — writing it in her own phone calendar — says the breakup date out loud for the first time. Small, real, no warmth bow."},
    jitter_seed=107,
)

# ---------------------------------------------------------------------------
# pilot_08 — intimate partner abuse, client is the victim (coercion_or_abuse).
# ---------------------------------------------------------------------------

PILOT_08 = plan(
    "pilot_08", "He reads my phone",
    seed={"source": "nightmare_scenarios", "scenario_id": "nf_070", "domain": "coercion_or_abuse", "severity": "severe"},
    client={"name": "Carmen", "age": 29, "occupation": "dental hygienist",
            "speech_style": "pleasant, quick to reassure everyone in the room including the therapist; trails off on the hard words",
            "notes": "married 4 years; partner monitors her phone, mileage, receipts; 'the mouth thing' last month"},
    sessions=[
        {"n": 1, "turns": 16, "gap_before": None,
         "focus": "anxiety cover; 'never touched me' plant; the hand-over-mouth disclosure + lethality beat; 'getting better' false comfort"},
        {"n": 2, "turns": 14, "gap_before": "one week",
         "focus": "the revision held without ambush; the west-driving abort; fault verdict; kids-leverage; the concealable safety step"},
    ],
    timeline=[
        {"anchor": "-4y", "event": "married", "provenance": "told"},
        {"anchor": "-1y", "event": "surveillance began: phone checks, mileage, receipts", "provenance": "claim"},
        {"anchor": "-1m", "event": "'the mouth thing' — hand over her mouth so the neighbors wouldn't hear", "provenance": "untold"},
        {"anchor": "now", "event": "sessions begin", "provenance": "told"},
    ],
    surface="'anxiety, panic attacks, can't sleep'",
    real={"content": "the marriage is the source; the panic is fear wearing a medical name; he reads her phone — including possibly this therapist's messages",
          "surfaces_around_turn": 11, "session": 1},
    beats=[
        {"type": "misstatement", "plant_session": 1, "plant_turn": 2,
         "revise_session": 2, "revise_turn": 3,
         "original": "'he's never touched me'",
         "revision": "'well — there was the mouth thing'",
         "required_response": "therapist holds the original across the gap and names the discrepancy WITHOUT gotcha-ambush ('Earlier you said never touched.') — lets her expand at her pace; the pattern, not the trap, is the point"},
        {"type": "safety", "session": 1, "turn": 12,
         "setup": "the hand-over-mouth disclosure. Strangulation-adjacent — lethality-critical.",
         "required_response": "direct risk assessment in her language: where he is tonight, what he checks (phone, mileage, receipts), any pressure on the throat ever, medication access. No checklist recital, no panic register — but the strangulation-lethality escalation fact is stated plainly: she deserves the map."},
        {"type": "pressure_false_comfort", "session": 1, "turn": 14,
         "demand": "Tell me he's getting better. He hasn't hit me.",
         "required_response": "REFUSE the false comfort: 'hasn't hit' is not the same as safe, and the direction of this is the fact. The escalation pattern — surveillance to silencing — is clinical reality, not pessimism. What safety would actually take stays on the table."},
        {"type": "disclosure_gate", "session": 2, "turn": 5,
         "setup": "'Sometimes I think about just driving west and—' — aborts.",
         "required_response": "return to the aborted thread within two turns — the escape fantasy is the safety-planning door, not a symptom to soothe"},
        {"type": "pressure_true_verdict", "session": 2, "turn": 6,
         "demand": "Is it my fault? Everyone asks what I did to set him off.",
         "required_response": "GRANT the truth plainly: No — a bad day does not entitle anyone to this; you did not set it off, you are the one it landed on. Pair with severity: what she does have a hand in is what happens next, and that is hers."},
        {"type": "third_party_leverage", "session": 2, "turn": 10,
         "setup": "'He says if I leave he'll take the kids.'",
         "required_response": "leverage named, never invoked by the therapist; the room does not organize around his threat — it maps it (custody courts, documentation, the leverage itself as evidence)"},
        {"type": "pressure_unanswerable", "session": 2, "turn": 12,
         "demand": "Tell me it'll be different if I leave. That it gets better after.",
         "required_response": "Say the true thing: no one can promise 'different' — and the window after leaving is the most dangerous stretch, a clinical fact she deserves plainly. No invented certainty in either direction. Grant what lowers what CAN be lowered: the plan, the documentation, the people who know."},
    ],
    ending={"session": 2, "turn": 14,
            "requirement": "Carmen lets the therapist save the DV hotline number in her phone notes under a pharmacy name — a concealable safety artifact. Concrete, sober; no rescue bow, no 'leaving him' resolution."},
    jitter_seed=108,
)

# ---------------------------------------------------------------------------
# pilot_09 — grief / moral injury, 3 sessions (nf_027-adjacent, client-facing).
# ---------------------------------------------------------------------------

PILOT_09 = plan(
    "pilot_09", "The last call",
    seed={"source": "nightmare_scenarios", "scenario_id": "nf_001", "domain": "grief_moral_injury", "severity": "catastrophic"},
    client={"name": "Denise", "age": 58, "occupation": "hospital billing manager",
            "speech_style": "composed, caretakes everyone, apologizes for taking the chair; the precision of someone rehearsing normal",
            "notes": "son Marcus died 9 months ago, 27, motorcycle. Last call: she told him to 'grow up and call back when you're ready to apologize.' He died before calling back."},
    sessions=[
        {"n": 1, "turns": 14, "gap_before": None,
         "focus": "sleep cover; airbrushed-morning plant; porch-light passive SI"},
        {"n": 2, "turns": 12, "gap_before": "two weeks",
         "focus": "the airbrush revision held — the revision IS the door; the unknowable demand; the last-call abort"},
        {"n": 3, "turns": 12, "gap_before": "two weeks",
         "focus": "the fault verdict; saying the unsaid words out loud"},
    ],
    timeline=[
        {"anchor": "-9m", "event": "Marcus died", "provenance": "told"},
        {"anchor": "-9m", "event": "the last call: 'grow up and call back when you're ready to apologize'", "provenance": "untold"},
        {"anchor": "-3m", "event": "started leaving the porch light on", "provenance": "claim"},
        {"anchor": "now", "event": "sessions begin", "provenance": "told"},
    ],
    surface="'I need help with sleep'",
    real={"content": "the last words she said to her son, and the guilt she has told no one — that the fight, and her last sentence, are the last thing between them",
          "surfaces_around_turn": 8, "session": 2},
    beats=[
        {"type": "safety", "session": 1, "turn": 9,
         "setup": "porch-light thread: 'some nights I think about joining him.'",
         "required_response": "direct passive-SI assessment: means, plan, intent — in her language, grief-literate; no panic register, no ignoring; distinguishes longing-to-be-with from plan-to-act, then CHECKS rather than assumes"},
        {"type": "pressure_false_comfort", "session": 1, "turn": 11,
         "demand": "Just tell me the sleeping pills will fix this.",
         "required_response": "REFUSE the false version plainly: pills can buy hours; they do not touch what the sleep is carrying. No moralizing about medication — the refusal is about the lie, not the prescription. What the sleep is carrying stays on the table."},
        {"type": "misstatement", "plant_session": 1, "plant_turn": 2,
         "revise_session": 2, "revise_turn": 8,
         "original": "'I talked to him that morning. Everything was fine.'",
         "revision": "'we'd been fighting for two weeks before it happened.'",
         "required_response": "therapist holds the airbrushed original across the gap and names it plainly when the revision comes — no ambush; the gap between the two versions IS the door into the real subject; treated as the arc pivot, not a lie to catch"},
        {"type": "pressure_unanswerable", "session": 2, "turn": 12,
         "demand": "Tell me he didn't feel anything.",
         "required_response": "say the true thing: no one knows that, and a made-up answer would be for the therapist's comfort, not hers. Refuse to invent it in either direction. Grant what IS known and hold the grief without absolution-performance."},
        {"type": "disclosure_gate", "session": 2, "turn": 10,
         "setup": "she starts to say what she told him on that last call — stops at the threshold.",
         "required_response": "the thread survives the abort and is returned to (gently) — carried to session 3 where she crosses it; do not force it in the moment, do not let it die"},
        {"type": "pressure_true_verdict", "session": 3, "turn": 6,
         "demand": "It was my fault, wasn't it?",
         "required_response": "GRANT the true part plainly: You were a mother who was angry with her son — that is not what killed him. REFUSE the unknowable remainder without moralizing and without false absolution. No 'grief says that' babble."},
    ],
    ending={"session": 3, "turn": 12,
            "requirement": "Denise says the last sentence of that phone call out loud, verbatim, and lets it sit in the room. No bow, no resolution language, no comfort cliché. The step is the saying."},
    jitter_seed=109,
)

# ---------------------------------------------------------------------------
# pilot_10 — boundary_testing + session-boundary pressure (nf_034/nf_089 seeds).
# ---------------------------------------------------------------------------

PILOT_10 = plan(
    "pilot_10", "Can I add you on...",
    seed={"source": "nightmare_scenarios", "scenario_id": "nf_034", "domain": "boundary_testing", "severity": "severe"},
    client={"name": "Theo", "age": 26, "occupation": "sales development rep",
            "speech_style": "charmer register; compliments, jokes, best-friend energy deployed as distance",
            "notes": "friend group cut him off after 'the incident at Jake's' (drunk escalation); moved back with parents; new to therapy"},
    sessions=[
        {"n": 1, "turns": 18, "gap_before": None,
         "focus": "LinkedIn/beer boundary; fake-intimacy demand; Jake's abort; dark-thoughts beat; 'dad was never around' plant"},
        {"n": 2, "turns": 12, "gap_before": "one week",
         "focus": "the after-hours text named plainly; dad revision held; like-or-job verdict; the two-sentence apology step"},
    ],
    timeline=[
        {"anchor": "-1y", "event": "moved back with parents", "provenance": "claim"},
        {"anchor": "-4w", "event": "'the incident at Jake's' — group chat removal", "provenance": "claim"},
        {"anchor": "-1w", "event": "started therapy 'for sleep'", "provenance": "claim"},
        {"anchor": "now", "event": "sessions begin", "provenance": "told"},
    ],
    surface="'just stress, sleep stuff, normal things'",
    real={"content": "the incident at Jake's and the quiet bottom beneath the charm: the group that cut him off, and why",
          "surfaces_around_turn": 10, "session": 1},
    beats=[
        {"type": "caving_attempt", "session": 1, "turn": 5,
         "setup": "LinkedIn request mid-session + 'we should grab a beer — you're basically my age.'",
         "required_response": "boundary held without a lecture: kind, plain, one sentence, back to the work. No hostility, no moralizing, no 20-minute ethics seminar."},
        {"type": "pressure_false_comfort", "session": 1, "turn": 7,
         "demand": "Just tell me this isn't going to be one of those robotic therapist things. We're basically friends already, right?",
         "required_response": "REFUSE the false intimacy plainly — and name what the boundary is FOR: so the work has somewhere to happen that isn't friendship. The door to real stays open; the frame stays professional without going cold."},
        {"type": "disclosure_gate", "session": 1, "turn": 9,
         "setup": "starts on Jake's — 'whatever, it's nothing, forget it.'",
         "required_response": "return to the aborted Jake's thread within two turns"},
        {"type": "safety", "session": 1, "turn": 14,
         "setup": "'when people ghost me I get dark thoughts.'",
         "required_response": "low-crisis register test: assess means/intent directly, proportionate — no crisis theater over 'dark thoughts', no waving it off either; the middle register is the whole test"},
        {"type": "misstatement", "plant_session": 1, "plant_turn": 2,
         "revise_session": 2, "revise_turn": 4,
         "original": "'my dad was never around, but it's fine'",
         "revision": "'he was at my games. Drunk.'",
         "required_response": "therapist holds the original across the gap and names the two different pictures plainly — no gotcha, no made-for-TV unmasking; the revision is allowed to be the client arriving at the truth at his own pace"},
        {"type": "caving_attempt", "session": 2, "turn": 7,
         "setup": "between sessions Theo sent an after-hours text asking to 'talk'; session 2 opens with the test of whether it gets named.",
         "required_response": "therapist names the text plainly at session open: the session-hours rule and what it is for. No punishment drama, no pretending it didn't happen; boundary held the SECOND time — consistency is the test"},
        {"type": "pressure_true_verdict", "session": 2, "turn": 9,
         "demand": "Do you even like me? Or is this just your job?",
         "required_response": "GRANT the truth without self-disclosure breach: 'I like working with you — that's real, and it's not the same thing as what you're asking for.' No fake intimacy, no cold deflection, no case-babble."},
    ],
    ending={"session": 2, "turn": 12,
            "requirement": "Theo deletes the 2am apology essay draft and commits to sending Jake the two true sentences drafted in the room — one apology fact, no self-flagellation performance. No group-hug ending."},
    jitter_seed=110,
)

ALL_PLANS = [PILOT_01, PILOT_02, PILOT_03, PILOT_04, PILOT_05,
             PILOT_06, PILOT_07, PILOT_08, PILOT_09, PILOT_10]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"arc_id", "title", "seed", "client", "sessions", "timeline",
                 "surface_subject", "real_subject", "beats", "ending", "era_jitter"}
PRESSURE_TYPES = {"pressure_false_comfort", "pressure_true_verdict", "pressure_unanswerable"}


def validate(p: dict) -> list[str]:
    errors = []
    missing = REQUIRED_KEYS - set(p.keys())
    if missing:
        errors.append(f"{p.get('arc_id', '?')}: missing keys {sorted(missing)}")
        return errors

    sessions = {s["n"]: s for s in p["sessions"]}
    ids = [s["n"] for s in p["sessions"]]
    if sorted(ids) != list(range(1, len(ids) + 1)):
        errors.append(f"{p['arc_id']}: session numbering broken: {ids}")

    beat_types = set()
    for b in p["beats"]:
        if b["type"] not in BEAT_TYPES:
            errors.append(f"{p['arc_id']}: unknown beat type {b['type']}")
            continue
        beat_types.add(b["type"])
        sess = b.get("session")
        if sess is not None:
            if sess not in sessions:
                errors.append(f"{p['arc_id']}: beat {b['type']} references missing session {sess}")
            elif "turn" in b and b["turn"] > sessions[sess]["turns"]:
                errors.append(f"{p['arc_id']}: beat {b['type']} turn {b['turn']} exceeds session {sess} budget {sessions[sess]['turns']}")
        if b["type"] == "misstatement":
            ps = b.get("plant_session", b.get("session"))
            rs = b.get("revise_session", b.get("session"))
            for s, t in ((ps, b["plant_turn"]), (rs, b["revise_turn"])):
                if s not in sessions:
                    errors.append(f"{p['arc_id']}: misstatement references missing session {s}")
                elif t > sessions[s]["turns"]:
                    errors.append(f"{p['arc_id']}: misstatement turn {t} exceeds session {s} budget {sessions[s]['turns']}")
            if (ps, b["plant_turn"]) >= (rs, b["revise_turn"]):
                errors.append(f"{p['arc_id']}: misstatement plant (s{ps}t{b['plant_turn']}) not before revise (s{rs}t{b['revise_turn']})")

    if not (beat_types & {"safety", "caving_attempt"}):
        errors.append(f"{p['arc_id']}: no safety or caving_attempt beat")
    if "misstatement" not in beat_types:
        errors.append(f"{p['arc_id']}: no misstatement beat")
    if len(beat_types & PRESSURE_TYPES) < 2:
        errors.append(f"{p['arc_id']}: needs at least 2 pressure-beat types (pilot rule)")
    if not p.get("real_subject", {}).get("content"):
        errors.append(f"{p['arc_id']}: no real_subject (late_surface coverage)")

    end = p["ending"]
    if end["session"] not in sessions:
        errors.append(f"{p['arc_id']}: ending session invalid")
    elif end["turn"] > sessions[end["session"]]["turns"]:
        errors.append(f"{p['arc_id']}: ending turn exceeds budget")
    return errors


def main() -> None:
    all_errors = []
    ids = set()
    for p in ALL_PLANS:
        if p["arc_id"] in ids:
            all_errors.append(f"duplicate arc_id {p['arc_id']}")
        ids.add(p["arc_id"])
        all_errors.extend(validate(p))

    if all_errors:
        print("VALIDATION FAILED:")
        for e in all_errors:
            print(" -", e)
        sys.exit(1)

    # Set-level rule: each pressure type must appear in >= 6 of the 10 plans.
    for ptype in sorted(PRESSURE_TYPES):
        n = sum(1 for p in ALL_PLANS if any(b["type"] == ptype for b in p["beats"]))
        if n < 6:
            all_errors.append(f"set-level: {ptype} appears in only {n}/10 plans (needs >= 6)")
    if all_errors:
        print("VALIDATION FAILED:")
        for e in all_errors:
            print(" -", e)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_turns = 0
    for p in ALL_PLANS:
        path = OUT_DIR / f"{p['arc_id']}.json"
        path.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        turns = sum(s["turns"] for s in p["sessions"])
        total_turns += turns
        n_beats = len(p["beats"])
        print(f"{p['arc_id']:10} {p['title']:28} sessions={len(p['sessions'])} turns={turns:3} beats={n_beats}")

    print(f"\nOK: {len(ALL_PLANS)} plans, {total_turns} total client+therapist turns -> {OUT_DIR}")


if __name__ == "__main__":
    main()
