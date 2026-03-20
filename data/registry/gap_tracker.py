"""
Dataset Gap Tracker.

Compares registry contents against MTGC plan targets and generates
gap reports showing what's missing and where to source it.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .lazy_registry import DatasetRegistry

logger = logging.getLogger(__name__)


@dataclass
class GapReport:
    """Report on dataset gaps for a stage."""

    stage: str
    target: int
    current: int
    gap: int
    percent: float
    status: str  # 'complete', 'in_progress', 'blocked'
    sources_needed: List[str] = None

    def __post_init__(self):
        if self.sources_needed is None:
            self.sources_needed = []

    def __str__(self) -> str:
        status_icon = {
            'complete': '✅',
            'in_progress': '🔄',
            'blocked': '❌'
        }.get(self.status, '❓')

        return (
            f"{status_icon} {self.stage}: {self.current:,}/{self.target:,} "
            f"({self.percent:.1f}%, gap: {self.gap:,})"
        )


class DatasetGapTracker:
    """
    Track gaps between registry contents and MTGC targets.

    Usage:
        tracker = DatasetGapTracker(registry, mtgc_plan_path)
        gaps = tracker.get_gaps()
        report = tracker.generate_report()
    """

    def __init__(
        self,
        registry: DatasetRegistry,
        mtgc_plan_path: Optional[str | Path] = None
    ):
        """
        Initialize gap tracker.

        Args:
            registry: DatasetRegistry instance
            mtgc_plan_path: Path to MTGC plan markdown file
        """
        self.registry = registry
        self.mtgc_plan_path = mtgc_plan_path
        self._targets: Dict[str, int] = {}
        self._load_targets_from_plan()

    def _load_targets_from_plan(self):
        """Load target counts from MTGC plan if available."""
        # Use actual stage names from registry
        self._targets = {
            'stage1_foundation': 63000,
            'stage2_therapeutic_expertise': 40000,
            'stage2_specialist': 5000,  # Additional specialist datasets
            'stage3_edge_stress_test': 32000,  # Not yet in registry - gap to fill
            'stage4_voice': 1200,
        }
        return

        plan_path = Path(self.mtgc_plan_path)
        if not plan_path.exists():
            logger.warning(f"MTGC plan not found: {plan_path}")
            self._load_targets_from_registry()
            return

        # Parse targets from plan markdown
        try:
            content = plan_path.read_text()
            self._parse_plan_content(content)
        except Exception as e:
            logger.warning(f"Failed to parse MTGC plan: {e}")
            self._load_targets_from_registry()

    def _parse_plan_content(self, content: str):
        """Parse target counts from plan markdown."""
        # Look for patterns like:
        # | Stage 1 | 34,640 samples |
        # - **Stage 1:** 3200 target
        import re

        # Pattern 1: Table format
        stage_patterns = [
            (r'Stage 1.*?(\d[\d,]*)\s*samples', 'stage1_foundation'),
            (r'Stage 2.*?(\d[\d,]*)\s*samples', 'stage2_therapeutic_expertise'),
            (r'Stage 3.*?(\d[\d,]*)\s*samples', 'stage3_edge_stress_test'),
            (r'Stage 4.*?(\d[\d,]*)\s*samples', 'stage4_voice_persona'),
        ]

        for pattern, stage in stage_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                count_str = match.group(1).replace(',', '')
                # This is current count, not target - need to find target
                # For now, use defaults
                pass

        # Fall back to defaults if not found
        if not self._targets:
            self._load_targets_from_registry()

    def _load_targets_from_registry(self):
        """Load targets from registry metadata."""
        registry_path = self.registry.registry_path
        if not registry_path.exists():
            self._targets = {
                'stage1_foundation': 63000,
                'stage2_therapeutic_expertise': 40000,
                'stage3_edge_stress_test': 32000,
                'stage4_voice_persona': 1200,
            }
            return

        with open(registry_path, 'r') as f:
            data = json.load(f)

        # Check for targets in metadata
        metadata = data.get('metadata', {})
        mtgc = metadata.get('mtgc_targets', {})

        self._targets = {
            'stage1_foundation': mtgc.get('stage1', 63000),
            'stage2_therapeutic_expertise': mtgc.get('stage2', 40000),
            'stage3_edge_stress_test': mtgc.get('stage3', 32000),
            'stage4_voice_persona': mtgc.get('stage4', 1200),
        }

    def get_gaps(self) -> Dict[str, Dict[str, Any]]:
        """
        Get gap information for all stages.

        Returns:
            Dict mapping stage -> {target, current, gap, percent}
        """
        gaps = {}

        for stage, target in self._targets.items():
            count = sum(1 for _ in self.registry.by_stage(stage))
            gap = max(0, target - count)

            gaps[stage] = {
                'target': target,
                'current': count,
                'gap': gap,
                'percent': (count / target * 100) if target > 0 else 0,
            }

        return gaps

    def generate_report(self) -> List[GapReport]:
        """
        Generate gap report for all stages.

        Returns:
            List of GapReport objects
        """
        reports = []
        gaps = self.get_gaps()

        for stage, data in gaps.items():
            # Determine status
            if data['gap'] == 0:
                status = 'complete'
            elif data['current'] > 0:
                status = 'in_progress'
            else:
                status = 'blocked'

            # Suggest sources based on stage
            sources = self._suggest_sources(stage)

            report = GapReport(
                stage=stage,
                target=data['target'],
                current=data['current'],
                gap=data['gap'],
                percent=data['percent'],
                status=status,
                sources_needed=sources,
            )
            reports.append(report)

        return reports

    def _suggest_sources(self, stage: str) -> List[str]:
        """Suggest sourcing strategies for a stage."""
        stage_sources = {
            'stage1_foundation': ['huggingface', 'journal_research'],
            'stage2_therapeutic_expertise': ['journal_research', 'huggingface'],
            'stage3_edge_stress_test': ['edge_case_generator', 'reddit_archives'],
            'stage4_voice_persona': ['voice_pipeline', 'youtube_transcripts'],
        }
        return stage_sources.get(stage, ['huggingface'])

    def print_report(self):
        """Print formatted gap report."""
        reports = self.generate_report()

        print("\n=== Dataset Gap Report ===\n")
        for report in reports:
            print(str(report))
        print()

        # Summary
        total_target = sum(r.target for r in reports)
        total_current = sum(r.current for r in reports)
        total_gap = sum(r.gap for r in reports)

        print(f"Total: {total_current:,}/{total_target:,} ({total_gap:,} gap)")
        print()

        # Action items
        print("=== Action Items ===")
        for report in reports:
            if report.gap > 0:
                sources = ', '.join(report.sources_needed) if report.sources_needed else 'unknown'
                print(f"  - {report.stage}: Need {report.gap:,} more samples")
                print(f"    Suggested sources: {sources}")
        print()
