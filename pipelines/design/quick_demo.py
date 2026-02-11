#!/usr/bin/env python3
"""Quick demo of NeMo Data Designer - generates data and shows results immediately."""

import sys
from ai.pipelines.design.service import NeMoDataDesignerService

def main():
    print("=" * 80)
    print("NVIDIA NeMo Data Designer - Quick Demo")
    print("=" * 80)
    print()

    service = NeMoDataDesignerService()

    print("Generating 10 synthetic therapeutic dataset samples...")
    print("(Using Preview API - instant, no job execution required)")
    print()

    try:
        result = service.generate_therapeutic_dataset(
            num_samples=10,
            include_demographics=True,
            include_symptoms=True,
            include_treatments=True,
            include_outcomes=True
        )

        print()
        print("✅ SUCCESS!")
        print("-" * 80)
        print(f"Generated: {result['num_samples']} samples")
        print(f"Time: {result['generation_time']:.2f} seconds")
        print(f"Columns: {len(result['column_names'])} columns")
        print()
        print("Column names:")
        for col in result['column_names']:
            print(f"  - {col}")
        print()

        # Display sample data
        dataset = result['data']
        if hasattr(dataset, 'to_pandas'):
            df = dataset.to_pandas()
            print("Sample Data (first 5 rows):")
            print("-" * 80)
            print(df.head(5).to_string())
            print()
            print(f"✅ Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        else:
            print("✅ Dataset generated successfully")
            print(f"Data type: {type(dataset)}")

        print()
        print("=" * 80)
        print("✅ NeMo Data Designer is working correctly!")
        print("=" * 80)
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

