import glob
import json
import os


def generate_analytics_report():
    chatml_files = glob.glob("ai/training/output/dataset/*.jsonl")
    if not chatml_files:
        print("No split files found in ai/training/output/dataset/.")
        return

    report = [
        "# Pixelated Empathy Dataset Card",
        "## Overview",
        "This dataset contains highly-curated, multi-turn clinical interactions explicitly tailored for therapeutic LLM fine-tuning.",
        "",
        "## Statistics"
    ]

    total_conversations = 0
    total_messages = 0
    total_words = 0

    split_stats = {}

    for file_path in chatml_files:
        filename = os.path.basename(file_path)
        file_convos = 0
        file_messages = 0
        file_words = 0

        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                pair = json.loads(line)
                file_convos += 1

                for msg in pair.get("messages", []):
                    file_messages += 1
                    file_words += len(msg.get("content", "").split())

        total_conversations += file_convos
        total_messages += file_messages
        total_words += file_words

        split_stats[filename] = {
            "conversations": file_convos,
            "messages": file_messages,
            "words": file_words,
            "avg_words_per_convo": round(file_words / file_convos, 2) if file_convos else 0
        }

    report.append(f"- **Total Conversations**: {total_conversations}")
    report.append(f"- **Total Messages**: {total_messages}")
    report.append(f"- **Total Estimated Words**: {total_words}")
    report.append(f"- **Average Words/Conversation**: {round(total_words / total_conversations, 2) if total_conversations else 0}")
    report.append("")
    report.append("## Split Breakdown")
    report.append("| Split | Conversations | Messages | Avg Words/Convo |")
    report.append("|---|---|---|---|")

    for split_name in sorted(split_stats.keys()):
        stats = split_stats[split_name]
        report.append(f"| {split_name} | {stats['conversations']} | {stats['messages']} | {stats['avg_words_per_convo']} |")

    report.append("")
    report.append("## Schema")
    report.append("Format: **ChatML** (`system`, `user`, `assistant`)")
    report.append("Privacy: **Fully scrubbed (Microsoft Presidio)**")
    report.append("Quality Gate: **LLM-Evaluated (Ollama Coherence Scorer >= 4)**")

    report_path = "ai/training/output/dataset/DATASET_CARD.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Dataset Card successfully generated at {report_path}")

if __name__ == "__main__":
    generate_analytics_report()
