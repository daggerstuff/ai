#!/usr/bin/env python3
"""Parse Whisper transcripts into speaker-separated OpenAI chat format.

Heuristic approach:
1. Detect and skip TV intro/outro narration.
2. Score each segment for therapist-likelihood.
3. Merge consecutive same-role segments into turns.
4. Output OpenAI chat format with system/assistant/user roles.
"""

import glob
import json
import os
import re
from pathlib import Path

IN_DIR = "/home/vivi/pixelated/data/youtube_mp3s/transcripts"
OUT_DIR = "/home/vivi/pixelated/data/youtube_mp3s/dialogues"
MERGED_OUT = "/home/vivi/pixelated/data/youtube_mp3s/phase2_media_training.jsonl"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


def therapist_score(text: str) -> float:
    t = text.strip().lower()
    if not t:
        return 0.0
    s = 0.0

    for pat in [
        r"^so\b", r"^why\b", r"^what\b", r"^where\b", r"^how\b",
        r"^did\b", r"^do\b", r"^tell\b", r"^when\b", r"^who\b",
        r"^let\'s\b", r"^let\b", r"^would\b", r"^could\b",
        r"^have\b", r"^are\b", r"^is\b", r"^was\b", r"^were\b", r"^can\b",
        r"^well,?\b", r"^that'?s\b", r"^you\b", r"^we\b",
    ]:
        if re.search(pat, t):
            s += 0.35

    if "?" in text:
        s += 0.25

    for w in [
        "experience", "therapy", "therapeutic", "session", "feelings",
        "emotion", "understand", "understanding", "awareness", "recognize",
        "pattern", "dynamic", "behavior", "unconscious", "conscious",
        "vulnerable", "vulnerability", "pain", "trauma", "relationship",
        "attachment", "boundary", "boundaries", "perspective", "reflect",
        "childhood", "family", "parent", "anxiety", "depression",
        "shame", "guilt", "heal", "healing", "validate", "empath",
        "insight", "realize", "connection", "connected", "disconnected",
        "comfortable", "uncomfortable", "honest", "authentic",
    ]:
        if w in t:
            s += 0.06

    if re.search(r"^(so you|so it|so that|what i hear|what i\'m hearing|it sounds like|i hear that|i understand that)", t):
        s += 0.4

    # Penalties
    if len(t) < 12 and "?" not in text:
        s -= 0.4
    if re.search(r"\b(i'm|i am|i don|i can|i have|i was|i think|i feel|i want|i need)\b", t):
        s -= 0.12
    if re.search(r"^(yeah|yes|no|nah|nope|mm\b|hmm\b|uh\b|uhhuh|okay|ok\b|right|sure|true|exactly|definitely|absolutely|totally|literally|like,|um,)", t):
        s -= 0.55

    return s


def is_intro_segment(text: str) -> bool:
    t = text.strip().lower()
    return any(
        m in t
        for m in [
            "my name is dr. siri", "my name is dr.siri",
            "this week i'm sitting down with", "sitting down with",
            "but today he is pushing", "but today she is pushing",
            "he's cozart", "also known as rapper",
            "17 year old rapper", "was arrested for", "sentenced to",
            "pointed a gun", "possession of marijuana",
            "watch more", "subscribe to", "follow us",
            "next time on", "this episode", "remember to", "check out our",
            "i'm jay shetty", "this podcast", "on this episode",
            "today on the podcast", "welcome to the",
        ]
    )


def extract_dialogue(filepath: str):
    with open(filepath) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    fname = Path(filepath).name

    if "Couples Therapy" in fname:
        show_type = "couples"
        therapist_name = "Dr. Orna Guralnik"
        fname.replace(".json", "").replace(" ｜ Couples Therapy", "").strip()
    elif "The Therapist" in fname:
        show_type = "therapist"
        therapist_name = "Dr. Siri Sat Nam Singh"
        fname.replace(".json", "").replace(" ｜ The Therapist", "").strip()
    else:
        return None, None, None

    if not segments:
        return None, None, None

    # 1. Detect intro range (Therapist only)
    intro_start = None
    intro_end = None
    if show_type == "therapist":
        for i, seg in enumerate(segments):
            if is_intro_segment(seg["text"]):
                if intro_start is None:
                    intro_start = i
                intro_end = i + 1
        if intro_start is not None and intro_end is None:
            intro_end = intro_start + 10

    # 2. Classify each segment
    scored = []
    for i, seg in enumerate(segments):
        if intro_start is not None and intro_start <= i < intro_end:
            continue
        text = seg["text"].strip()
        if not text:
            continue
        score = therapist_score(text)
        tl = text.lower()
        if "dr. orna" in tl or "dr orna" in tl or "dr. siri" in tl or "dr siri" in tl:
            score = 10.0
        if is_intro_segment(text):
            score = -10.0
        scored.append({"start": seg["start"], "text": text, "score": score})

    if not scored:
        return None, None, None

    # 3. Initial role assignment per-segment
    roles = ["therapist" if s["score"] > 0.12 else "client" for s in scored]

    # 4. Smoothing: fix isolated single-segment misclassifications
    for i in range(1, len(roles) - 1):
        prev_r = roles[i - 1]
        next_r = roles[i + 1]
        curr_r = roles[i]
        if curr_r not in (prev_r, next_r) and prev_r == next_r and abs(scored[i]["score"]) <= 1.0:
            roles[i] = prev_r

    # 5. Merge consecutive same-role into turns
    messages = [{"role": "system", "content": f"You are {therapist_name}. This is a therapy session."}]
    current_role = None
    current_text = ""
    therapist_turns = 0
    client_turns = 0

    for item, role in zip(scored, roles, strict=True):
        if role == "skip":
            continue
        if role == current_role:
            current_text += " " + item["text"]
        else:
            if current_role is not None:
                oai_role = "assistant" if current_role == "therapist" else "user"
                messages.append({"role": oai_role, "content": current_text.strip()})
            current_role = role
            current_text = item["text"]

    if current_role is not None:
        oai_role = "assistant" if current_role == "therapist" else "user"
        messages.append({"role": oai_role, "content": current_text.strip()})

    for m in messages:
        if m["role"] == "assistant":
            therapist_turns += 1
        elif m["role"] == "user":
            client_turns += 1

    if len(messages) < 3:
        return None, None, None

    return messages, therapist_turns, client_turns


def main():
    files = sorted(glob.glob(f"{IN_DIR}/*.json"))
    all_records = []
    stats = []

    for fp in files:
        fname = Path(fp).name
        messages, t_turns, c_turns = extract_dialogue(fp)
        if messages is None or t_turns is None or c_turns is None:
            stats.append({"file": fname, "status": "failed"})
            continue

        dialogue_turns = t_turns + c_turns
        out_path = os.path.join(OUT_DIR, fname.replace(".json", "_dialogue.json"))
        with open(out_path, "w") as f:
            json.dump(
                {
                    "source_file": fname,
                    "messages": messages,
                    "stats": {
                        "therapist_turns": t_turns,
                        "client_turns": c_turns,
                        "total_turns": dialogue_turns,
                    },
                },
                f,
                indent=2,
            )

        all_records.append({"source_file": fname, "messages": messages})
        stats.append(
            {
                "file": fname,
                "status": "ok",
                "therapist_turns": t_turns,
                "client_turns": c_turns,
                "total_turns": dialogue_turns,
            }
        )

    with open(MERGED_OUT, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    ok_count = sum(1 for s in stats if s["status"] == "ok")
    fail_count = sum(1 for s in stats if s["status"] == "failed")
    total_t = sum(s.get("therapist_turns", 0) for s in stats if s["status"] == "ok")
    total_c = sum(s.get("client_turns", 0) for s in stats if s["status"] == "ok")
    total = sum(s.get("total_turns", 0) for s in stats if s["status"] == "ok")


    stats_path = MERGED_OUT.replace(".jsonl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(
            {
                "total_files": len(files),
                "success": ok_count,
                "failed": fail_count,
                "total_therapist_turns": total_t,
                "total_client_turns": total_c,
                "total_dialogue_turns": total,
                "per_file": stats,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
