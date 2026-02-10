"""
Multi-Agent Annotation Runner
Execute multi-agent annotation with consensus building
"""

import argparse
import json
import time
from pathlib import Path

from multi_agent_system import create_multi_agent_system


def process_batch_multi_agent(
    input_file: str,
    output_file: str,
    model: str = "nvidia/nemotron-3-nano-30b-a3b",
):
    """
    Process a batch file with multi-agent annotation
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    print("🤖 Multi-Agent Annotation System")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print(f"🧠 Model: {model}")
    print("-" * 60)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    # Create multi-agent system
    orchestrator = create_multi_agent_system(model=model)
    print(f"✅ Initialized {len(orchestrator.agents)} agents")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process batch
    processed_count = 0
    total_processing_time = 0

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            try:
                task = json.loads(line)
                start_time = time.time()

                # Run multi-agent annotation
                result = orchestrator.annotate_with_consensus(task)

                processing_time = time.time() - start_time
                total_processing_time += processing_time

                # Write result
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
                processed_count += 1

                # Progress update
                if processed_count % 10 == 0:
                    avg_time = total_processing_time / processed_count
                    print(
                        f"  📊 Processed {processed_count} | Avg time: {avg_time:.2f}s",
                        end="\r",
                    )

            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping invalid JSON: {e}")
                continue
            except Exception as e:
                print(f"❌ Error processing task: {e}")
                continue

    # Final summary
    print("\n" + "=" * 60)
    print(f"✅ Completed: {processed_count} annotations")
    print(f"⏱️  Total time: {total_processing_time:.2f}s")
    print(f"📈 Average time per annotation: {total_processing_time / processed_count:.2f}s")
    print(f"💾 Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Agent Annotation System")
    parser.add_argument(
        "--input",
        required=True,
        help="Input batch JSONL file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output results JSONL file",
    )
    parser.add_argument(
        "--model",
        default="nvidia/nemotron-3-nano-30b-a3b",
        help="LLM model to use (default: nvidia/nemotron-3-nano-30b-a3b)",
    )

    args = parser.parse_args()

    process_batch_multi_agent(
        input_file=args.input,
        output_file=args.output,
        model=args.model,
    )
