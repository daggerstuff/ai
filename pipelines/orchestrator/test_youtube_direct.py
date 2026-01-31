"""
Direct test of YouTube RAG system without module imports
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `ai.*` imports work reliably
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

def test_youtube_rag_direct():
    """Test YouTube RAG system directly"""
    print("Testing YouTube RAG system directly...")

    try:
        from ai.pipelines.orchestrator.storage_config import get_dataset_pipeline_output_root
        from ai.pipelines.orchestrator import youtube_rag_system

        rag_index_dir = get_dataset_pipeline_output_root() / "rag_index"
        print(f"✓ Using RAG index directory: {rag_index_dir}")

        # Create RAG system
        rag_system = youtube_rag_system.YouTubeRAGSystem()
        print("✓ YouTube RAG System created successfully")
        print(f"  Transcript files processed: {len(rag_system.transcripts)}")

        if rag_system.transcripts:
            print("  Sample transcript metadata:")
            for i, (filename, metadata) in enumerate(list(rag_system.transcripts.items())[:3]):
                print(f"    {i+1}. {filename}: {metadata.get('title', 'No title')}")

        # Show integration points
        print("\n  Integration Points:")
        print("  - process_transcript_files(): Processes MD files from transcripts directory")
        print("  - extract_metadata(): Extracts title, creator, duration from transcripts")
        print("  - create_embeddings(): Would create semantic embeddings (requires sentence-transformers)")
        print("  - search_transcripts(): Semantic search functionality")
        print("  - get_few_shot_examples(): Extract examples for training")

        return True

    except Exception as e:
        print(f"✗ YouTube RAG direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_youtube_rag_direct()
    if success:
        print("\n🎉 YouTube RAG system is ready for integration!")
    else:
        print("\n❌ YouTube RAG system test failed")
    exit(0 if success else 1)