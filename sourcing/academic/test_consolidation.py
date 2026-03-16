#!/usr/bin/env python3
"""
Test script for consolidated Academic Sourcing Engine

This script verifies that the consolidation was successful and all
components are working correctly.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.sourcing.academic import (
    AcademicSourcingEngine,
    BookMetadata,
    SourcingStrategy,
    SourceType,
    create_academic_sourcing_engine,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all imports work"""
    logger.info("✅ All imports successful")
    return True


def test_engine_creation():
    """Test engine creation with different strategies"""
    try:
        # Test default (HYBRID)
        engine1 = AcademicSourcingEngine()
        assert engine1.strategy == SourcingStrategy.HYBRID
        logger.info("✅ Default HYBRID engine created")

        # Test factory function
        engine2 = create_academic_sourcing_engine(strategy="api_only")
        assert engine2.strategy == SourcingStrategy.API_ONLY
        logger.info("✅ Factory function works")

        # Test all strategies
        for strategy in SourcingStrategy:
            engine = AcademicSourcingEngine(strategy=strategy)
            assert engine.strategy == strategy
        logger.info("✅ All strategies work")

        return True
    except Exception as e:
        logger.error(f"❌ Engine creation failed: {e}")
        return False


def test_metadata_structure():
    """Test BookMetadata dataclass"""
    try:
        metadata = BookMetadata(
            title="Test Book",
            authors=["Author One", "Author Two"],
            publisher="Test Publisher",
            publication_year=2024,
            source="test",
        )

        # Test to_dict conversion
        data_dict = metadata.to_dict()
        assert data_dict["title"] == "Test Book"
        assert len(data_dict["authors"]) == 2

        logger.info("✅ BookMetadata structure works")
        return True
    except Exception as e:
        logger.error(f"❌ Metadata test failed: {e}")
        return False


def test_api_sources():
    """Test API source methods (without actually calling APIs)"""
    try:
        engine = AcademicSourcingEngine()

        # Verify methods exist
        assert hasattr(engine, "fetch_arxiv_papers")
        assert hasattr(engine, "fetch_semantic_scholar")
        assert hasattr(engine, "fetch_crossref")
        assert hasattr(engine, "scrape_google_scholar")

        logger.info("✅ All API source methods exist")
        return True
    except Exception as e:
        logger.error(f"❌ API sources test failed: {e}")
        return False


def test_publisher_integration():
    """Test publisher integration framework"""
    try:
        engine = AcademicSourcingEngine()

        # Check APA publisher is initialized (if API key is available)
        if SourceType.APA_PUBLISHER in engine.publishers:
            logger.info("✅ APA publisher initialized (API key found)")
        else:
            logger.info("ℹ️  APA publisher not initialized (no API key)")

        # Verify method exists
        assert hasattr(engine, "fetch_from_publisher")

        logger.info("✅ Publisher integration framework works")
        return True
    except Exception as e:
        logger.error(f"❌ Publisher integration test failed: {e}")
        return False


def test_utility_methods():
    """Test utility methods"""
    try:
        engine = AcademicSourcingEngine()

        # Test therapeutic relevance scoring
        metadata = BookMetadata(
            title="Cognitive Behavioral Therapy for Depression",
            authors=["Test Author"],
            publisher="Test Publisher",
            publication_year=2024,
            source="test",
            abstract="This book covers CBT techniques for treating depression and anxiety.",
        )

        score = engine._score_therapeutic_relevance(metadata)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should score high due to keywords

        logger.info(f"✅ Therapeutic relevance scoring works (score: {score:.2f})")

        # Test deduplication
        results = [
            BookMetadata(
                title="Test Book",
                authors=["Author One"],
                publisher="Pub",
                publication_year=2024,
                source="test1",
            ),
            BookMetadata(
                title="Test Book",
                authors=["Author One"],
                publisher="Pub",
                publication_year=2024,
                source="test2",
            ),  # Duplicate
            BookMetadata(
                title="Different Book",
                authors=["Author Two"],
                publisher="Pub",
                publication_year=2024,
                source="test3",
            ),
        ]

        deduplicated = engine._deduplicate_results(results)
        assert len(deduplicated) == 2

        logger.info("✅ Deduplication works")
        return True
    except Exception as e:
        logger.error(f"❌ Utility methods test failed: {e}")
        return False


def test_export():
    """Test export functionality"""
    try:
        engine = AcademicSourcingEngine()

        # Create test data
        test_data = [
            BookMetadata(
                title="Test Book 1",
                authors=["Author One"],
                publisher="Test Publisher",
                publication_year=2024,
                source="test",
            ),
            BookMetadata(
                title="Test Book 2",
                authors=["Author Two"],
                publisher="Test Publisher",
                publication_year=2024,
                source="test",
            ),
        ]

        # Export to temp file
        output_path = engine.export_data(test_data, "test_export.json")
        assert output_path.exists()

        # Clean up
        output_path.unlink()

        logger.info("✅ Export functionality works")
        return True
    except Exception as e:
        logger.error(f"❌ Export test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Testing Consolidated Academic Sourcing Engine")
    logger.info("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Engine Creation", test_engine_creation),
        ("Metadata Structure", test_metadata_structure),
        ("API Sources", test_api_sources),
        ("Publisher Integration", test_publisher_integration),
        ("Utility Methods", test_utility_methods),
        ("Export", test_export),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing: {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 All tests passed! Consolidation successful!")
        return 0
    else:
        logger.error(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
