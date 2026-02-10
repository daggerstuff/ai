#!/usr/bin/env python3
"""
CPTSD Source Catalog and Download Script

This script catalogs and downloads CPTSD-related content from S3 for dataset expansion.
It identifies all CPTSD sources, downloads them, and creates a comprehensive inventory.
"""

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CPTSDSource:
    """Represents a CPTSD content source."""
    source_name: str
    s3_key: str
    file_size: int
    estimated_topics: List[str]
    content_type: str  # transcript, article, book, etc.
    voice_profile: Optional[str] = None


class CPTSDSourceCataloger:
    """Catalogs and manages CPTSD data sources."""

    # CPTSD-related keywords for identification
    CPTSD_KEYWORDS = [
        'cptsd', 'complex ptsd', 'complex trauma', 'complex_trauma',
        'emotional flashback', 'emotional_flashback', 'flashback',
        'shame cycle', 'shame_cycle', 'inner child', 'inner_child',
        'reparenting', 're-parenting', 'trauma bonding', 'trauma_bonding',
        'betrayal trauma', 'betrayal_trauma', 'survival mode', 'survival_mode',
        'hypervigilance', 'hyper-vigilance', 'dissociation',
        'emotional dysregulation', 'emotional_dysregulation',
        'toxic shame', 'toxic_shame', 'procrastination shame',
        'pete walker', 'heidi priebe', 'patrick teahan'
    ]

    # Known CPTSD sources from S3 manifest
    KNOWN_SOURCES = {
        'heidi_priebe': {
            'prefix': 'datasets/gdrive/tier4_voice_persona/Heidi Priebe/',
            'voice_profile': 'heidi_priebe',
            'content_type': 'article'
        },
        'patrick_teahan': {
            'prefix': 'datasets/gdrive/tier4_voice_persona/Patrick Teahan /',
            'voice_profile': 'patrick_teahan',
            'content_type': 'transcript'
        },
        'tim_fletcher': {
            'prefix': 'datasets/gdrive/tier4_voice_persona/Tim Fletcher/',
            'voice_profile': 'tim_fletcher',
            'content_type': 'transcript'
        },
        'pete_walker': {
            'prefix': 'datasets/gdrive/knowledge/books/',
            'voice_profile': None,
            'content_type': 'book'
        }
    }

    def __init__(self, output_dir: str = '/tmp/cptsd_sources'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_file = self.output_dir / 'cptsd_source_catalog.json'
        self.sources: List[CPTSDSource] = []

    def run_ovhai_command(self, command: List[str]) -> str:
        """Execute an ovhai CLI command and return output."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(command)}")
            print(f"Error: {e.stderr}")
            return ""

    def list_s3_objects(self, prefix: str) -> List[Dict]:
        """List objects in S3 with given prefix."""
        output = self.run_ovhai_command([
            'ovhai', 'bucket', 'object', 'list',
            'pixel-data@US-EAST-VA',
            '--prefix', prefix
        ])

        objects = []
        for line in output.strip().split('\n'):
            if line.strip():
                # Parse line: key size date
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0]
                    try:
                        size = int(parts[1])
                    except (ValueError, IndexError):
                        size = 0
                    objects.append({'key': key, 'size': size})

        return objects

    def identify_cptsd_content(self, filename: str) -> List[str]:
        """Identify CPTSD topics based on filename."""
        topics = []
        filename_lower = filename.lower()

        # Topic mapping
        topic_keywords = {
            'emotional_flashbacks': ['flashback', 'flashbacks', 'emotional flashback'],
            'shame_cycles': ['shame', 'shame cycle', 'toxic shame', 'procrastination shame'],
            'inner_child': ['inner child', 'inner_child', 'reparenting', 're-parenting'],
            'trauma_bonding': ['trauma bonding', 'trauma_bonding', 'bonding'],
            'betrayal_trauma': ['betrayal trauma', 'betrayal_trauma', 'betrayal'],
            'survival_mode': ['survival mode', 'survival_mode', 'survival'],
            'hypervigilance': ['hypervigilance', 'hyper-vigilance', 'vigilance'],
            'dissociation': ['dissociation', 'dissociate'],
            'emotional_dysregulation': ['dysregulation', 'emotional dysregulation', 'regulation'],
            'recovery': ['recovery', 'healing', 'heal', 'thriving'],
            'triggers': ['trigger', 'triggers'],
            'boundaries': ['boundary', 'boundaries'],
            'codependency': ['codependency', 'codependent', 'co-narcissist'],
            'narcissism': ['narcissist', 'narcissism', 'narcissistic'],
            'anger': ['anger', 'angry'],
            'relationships': ['relationship', 'relationships', 'attachment'],
            'self_care': ['self care', 'self-care', 'depletion'],
            'sexuality': ['sex', 'sexuality', 'intimacy']
        }

        topics.extend(
            topic
            for topic, keywords in topic_keywords.items()
            if any(keyword in filename_lower for keyword in keywords)
        )
        return topics

    def catalog_heidi_priebe(self) -> List[CPTSDSource]:
        """Catalog Heidi Priebe CPTSD content."""
        print("Cataloging Heidi Priebe CPTSD content...")
        sources = []

        prefix = self.KNOWN_SOURCES['heidi_priebe']['prefix']
        objects = self.list_s3_objects(prefix)

        for obj in objects:
            filename = obj['key'].split('/')[-1]
            # Filter for CPTSD-related content
            if any(keyword in filename.lower() for keyword in ['cptsd', 'complex', 'shame', 'survival']):
                topics = self.identify_cptsd_content(filename)
                sources.append(CPTSDSource(
                    source_name='heidi_priebe',
                    s3_key=obj['key'],
                    file_size=obj['size'],
                    estimated_topics=topics,
                    content_type='article',
                    voice_profile='heidi_priebe'
                ))

        print(f"Found {len(sources)} Heidi Priebe CPTSD files")
        return sources

    def catalog_patrick_teahan(self) -> List[CPTSDSource]:
        """Catalog Patrick Teahan CPTSD content."""
        print("Cataloging Patrick Teahan CPTSD content...")
        sources = []

        prefix = self.KNOWN_SOURCES['patrick_teahan']['prefix']
        objects = self.list_s3_objects(prefix)

        for obj in objects:
            filename = obj['key'].split('/')[-1]
            # Filter for CPTSD-related content
            if any(keyword in filename.lower() for keyword in ['cptsd', 'ptsd', 'shame', 'trauma']):
                topics = self.identify_cptsd_content(filename)
                sources.append(CPTSDSource(
                    source_name='patrick_teahan',
                    s3_key=obj['key'],
                    file_size=obj['size'],
                    estimated_topics=topics,
                    content_type='transcript',
                    voice_profile='patrick_teahan'
                ))

        print(f"Found {len(sources)} Patrick Teahan CPTSD files")
        return sources

    def catalog_tim_fletcher(self) -> List[CPTSDSource]:
        """Catalog Tim Fletcher complex trauma content."""
        print("Cataloging Tim Fletcher complex trauma content...")
        sources = []

        prefix = self.KNOWN_SOURCES['tim_fletcher']['prefix']
        objects = self.list_s3_objects(prefix)

        for obj in objects:
            filename = obj['key'].split('/')[-1]
            # Tim Fletcher has extensive complex trauma content
            if any(keyword in filename.lower() for keyword in ['complex', 'trauma', 'shame', 'characteristics', 'recovery']):
                topics = self.identify_cptsd_content(filename)
                sources.append(CPTSDSource(
                    source_name='tim_fletcher',
                    s3_key=obj['key'],
                    file_size=obj['size'],
                    estimated_topics=topics,
                    content_type='transcript',
                    voice_profile='tim_fletcher'
                ))

        print(f"Found {len(sources)} Tim Fletcher complex trauma files")
        return sources

    def catalog_pete_walker(self) -> List[CPTSDSource]:
        """Catalog Pete Walker book content."""
        print("Cataloging Pete Walker book content...")
        sources = []

        # Look for Pete Walker's book
        prefix = self.KNOWN_SOURCES['pete_walker']['prefix']
        objects = self.list_s3_objects(prefix)

        for obj in objects:
            filename = obj['key'].split('/')[-1]
            if 'pete walker' in filename.lower() or 'complex ptsd' in filename.lower():
                sources.append(CPTSDSource(
                    source_name='pete_walker',
                    s3_key=obj['key'],
                    file_size=obj['size'],
                    estimated_topics=['comprehensive_cptsd_guide', 'emotional_flashbacks', 'recovery_stages'],
                    content_type='book',
                    voice_profile=None
                ))

        print(f"Found {len(sources)} Pete Walker book files")
        return sources

    def catalog_all_sources(self) -> None:
        """Catalog all CPTSD sources."""
        print("=" * 60)
        print("CPTSD Source Cataloging")
        print("=" * 60)

        self.sources.extend(self.catalog_heidi_priebe())
        self.sources.extend(self.catalog_patrick_teahan())
        self.sources.extend(self.catalog_tim_fletcher())
        self.sources.extend(self.catalog_pete_walker())

        print(f"\nTotal CPTSD sources found: {len(self.sources)}")

    def save_catalog(self) -> None:
        """Save catalog to JSON file."""
        catalog_data = {
            'generated_at': datetime.now().isoformat(),
            'total_sources': len(self.sources),
            'sources_by_type': {},
            'sources': [asdict(source) for source in self.sources]
        }

        # Group by source name
        for source in self.sources:
            if source.source_name not in catalog_data['sources_by_type']:
                catalog_data['sources_by_type'][source.source_name] = 0
            catalog_data['sources_by_type'][source.source_name] += 1

        with open(self.catalog_file, 'w') as f:
            json.dump(catalog_data, f, indent=2)

        print(f"\nCatalog saved to: {self.catalog_file}")

    def print_summary(self) -> None:
        """Print catalog summary."""
        print("\n" + "=" * 60)
        print("CPTSD Source Summary")
        print("=" * 60)

        # Group by source
        by_source: Dict[str, List[CPTSDSource]] = {}
        for source in self.sources:
            if source.source_name not in by_source:
                by_source[source.source_name] = []
            by_source[source.source_name].append(source)

        for source_name, sources in sorted(by_source.items()):
            total_size = sum(s.file_size for s in sources)
            print(f"\n{source_name.upper().replace('_', ' ')}:")
            print(f"  Files: {len(sources)}")
            print(f"  Total Size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")

            # Show unique topics
            all_topics = set()
            for s in sources:
                all_topics.update(s.estimated_topics)
            print(f"  Topics: {', '.join(sorted(all_topics))}")

        # Overall statistics
        total_size = sum(s.file_size for s in self.sources)
        all_topics = set()
        for s in self.sources:
            all_topics.update(s.estimated_topics)

        print("\nOVERALL:")
        print(f"  Total Files: {len(self.sources)}")
        print(f"  Total Size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
        print(f"  Unique Topics: {len(all_topics)}")
        print(f"  Topics: {', '.join(sorted(all_topics))}")


def main():
    """Main entry point."""
    cataloger = CPTSDSourceCataloger()

    # Catalog all sources
    cataloger.catalog_all_sources()

    # Save catalog
    cataloger.save_catalog()

    # Print summary
    cataloger.print_summary()

    print("\n" + "=" * 60)
    print("Cataloging complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
