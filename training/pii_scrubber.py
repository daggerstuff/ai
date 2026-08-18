import glob
import json
import os

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


def main():
    print("Loading Presidio Analyzer...")
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    chatml_files = glob.glob("ai/training/output/books/chatml/*.jsonl")
    if not chatml_files:
        print("No ChatML files found to scan.")
        return

    print(f"Found {len(chatml_files)} files. Starting PII scan...")
    total_pairs = 0
    total_redactions = 0

    # Entities we care about for therapeutic privacy
    # Removing 'PERSON' by default because books naturally have character names,
    # but we will scan for highly sensitive things like emails, phones, locations.
    sensitive_entities = [
        "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
        "CRYPTO", "IP_ADDRESS", "MEDICAL_LICENSE", "US_SSN", "US_PASSPORT"
    ]

    for file_path in chatml_files:
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()

        clean_lines = []
        file_redactions = 0

        for line in lines:
            try:
                pair = json.loads(line)
                total_pairs += 1
                messages = pair.get("messages", [])

                # Scan each message in the ChatML conversation
                for msg in messages:
                    text = msg.get("content", "")

                    # Analyze text for PII
                    results = analyzer.analyze(
                        text=text,
                        entities=sensitive_entities,
                        language='en'
                    )

                    if results:
                        # Anonymize
                        anonymized_result = anonymizer.anonymize(
                            text=text,
                            analyzer_results=results
                        )
                        msg["content"] = anonymized_result.text
                        file_redactions += len(results)

                clean_lines.append(json.dumps(pair))
            except Exception as e:
                print(f"Error processing line: {e}")

        # If we found PII, overwrite the file with the scrubbed version
        if file_redactions > 0:
            print(f"Found and scrubbed {file_redactions} PII instances in {os.path.basename(file_path)}")
            total_redactions += file_redactions
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(clean_lines) + "\n")
        else:
            print(f"Clean (0 PII instances): {os.path.basename(file_path)}")

    print("=" * 40)
    print("PII Scan Complete.")
    print(f"Total Conversations Scanned: {total_pairs}")
    print(f"Total Redactions Made: {total_redactions}")

if __name__ == "__main__":
    main()
