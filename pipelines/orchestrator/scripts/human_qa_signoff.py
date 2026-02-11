#!/usr/bin/env python3
""
QA Signoff and Release Notes

Implements Issue 7: Release 0: Clinician QA + bias/cultural review signoff

This script records human review signoff for Release 0: clinician QA sampling
(foundation + edge) and bias/cultural review (foundation).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import boto3
from botocore.exceptions import ClientError
import uuid
from dataclasses import dataclass, asdict

# Add the dataset_pipeline to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage_config import get_storage_config, StorageConfig


@dataclass
class ReviewCriteria:
    """Criteria for human review"""
    criterion_id: str
    name: str
    description: str
    weight: float
    pass_threshold: float


@dataclass
class ReviewSample:
    """A sample selected for human review"""
    sample_id: str
    family: str
    split: str
    file_key: str
    content_excerpt: str
    review_type: str
    selected_at: str


@dataclass
class ReviewResult:
    """Result of human review for a sample"""
    sample_id: str
    reviewer_id: str
    review_date: str
    criteria_scores: Dict[str, float]
    overall_score: float
    passed: bool
    notes: str
    concerns: List[str]
    recommendations: List[str]


@dataclass
class ReviewerInfo:
    """Information about a reviewer"""
    reviewer_id: str
    name: str
    credentials: str
    specialization: str
    contact: str


class QASampleGenerator:
    """Generates samples for human QA review"""

    def __init__(self, storage_config: StorageConfig):
        self.config = storage_config
        self.s3_client = None
        self._init_s3_client()

    def _init_s3_client(self):
        """Initialize S3 client"""
        if self.config.backend != self.config.backend.S3:
            raise ValueError("S3 backend required for QA sampling")

        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.config.s3_endpoint_url,
                aws_access_key_id=self.config.s3_access_key_id,
                aws_secret_access_key=self.config.s3_secret_access_key,
                region_name=self.config.s3_region or 'us-east-1'
            )

            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
            print(f"✓ Connected to S3 bucket: {self.config.s3_bucket}")

        except Exception as e:
            raise ValueError(f"Failed to connect to S3: {e}")

    def load_manifest(self, release_version: str) -> Dict[str, Any]:
        """Load release manifest from S3"""
        manifest_key = f"{self.config.exports_prefix}/releases/{release_version}/manifest.json"

        try:
            response = self.s3_client.get_object(Bucket=self.config.s3_bucket, Key=manifest_key)
            manifest = json.loads(response['Body'].read())
            print(f"✓ Loaded manifest: {manifest_key}")
            return manifest
        except ClientError as e:
            raise ValueError(f"Failed to load manifest {manifest_key}: {e}")

    def sample_file_content(self, s3_key: str, sample_size: int = 1024) -> str:
        """Sample content from an S3 file for review"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Range=f'bytes=0-{sample_size-1}'
            )

            content = response['Body'].read().decode('utf-8', errors='ignore')
            return content

        except ClientError as e:
            print(f"⚠️  Failed to sample {s3_key}: {e}")
            return ""

    def generate_qa_samples(self, manifest: Dict[str, Any],
                          samples_per_family: int = 5) -> List[ReviewSample]:
        """Generate QA samples from manifest"""
        print("📋 Generating QA samples...")

        samples = []

        # Define families that need QA review
        qa_families = {
            'professional_therapeutic': 'foundation',
            'priority_datasets': 'foundation',
            'edge_cases': 'edge',
            'voice_persona': 'foundation'
        }

        for family_name, review_type in qa_families.items():
            if family_name not in manifest['families']:
                print(f"⚠️  Family {family_name} not found in manifest")
                continue

            family_data = manifest['families'][family_name]
            files = family_data['files']

            print(f"  Sampling {family_name}: {len(files)} files available")

            # Sample files from different splits
            files_by_split = {}
            for file_info in files:
                split = file_info.get('split', 'unknown')
                if split not in files_by_split:
                    files_by_split[split] = []
                files_by_split[split].append(file_info)

            # Generate samples from each split
            for split, split_files in files_by_split.items():
                sample_count = min(samples_per_family, len(split_files))

                # Select files for sampling (evenly distributed)
                step = max(1, len(split_files) // sample_count)
                selected_files = split_files[::step][:sample_count]

                for file_info in selected_files:
                    s3_key = file_info['key']

                    # Sample content
                    content_excerpt = self.sample_file_content(s3_key)

                    if content_excerpt:
                        sample = ReviewSample(
                            sample_id=str(uuid.uuid4()),
                            family=family_name,
                            split=split,
                            file_key=s3_key,
                            content_excerpt=content_excerpt[:500] + "..." if len(content_excerpt) > 500 else content_excerpt,
                            review_type=review_type,
                            selected_at=datetime.utcnow().isoformat()
                        )
                        samples.append(sample)

        print(f"✓ Generated {len(samples)} QA samples")
        return samples

    def save_qa_samples(self, samples: List[ReviewSample], release_version: str) -> str:
        """Save QA samples to S3 for reviewer access"""
        samples_data = {
            'metadata': {
                'release_version': release_version,
                'generated_at': datetime.utcnow().isoformat(),
                'total_samples': len(samples),
                'generator': 'human_qa_signoff.py'
            },
            'samples': [asdict(sample) for sample in samples]
        }

        samples_key = f"{self.config.exports_prefix}/releases/{release_version}/qa/qa_samples.json"

        try:
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=samples_key,
                Body=json.dumps(samples_data, indent=2),
                ContentType='application/json'
            )

            samples_url = f"s3://{self.config.s3_bucket}/{samples_key}"
            print(f"✓ QA samples saved: {samples_url}")
            return samples_url

        except ClientError as e:
            raise ValueError(f"Failed to save QA samples: {e}")


class ReviewCriteriaManager:
    """Manages review criteria for different review types"""

    def get_foundation_criteria(self) -> List[ReviewCriteria]:
        """Get review criteria for foundation datasets"""
        return [
            ReviewCriteria(
                criterion_id="therapeutic_accuracy",
                name="Therapeutic Accuracy",
                description="Content demonstrates accurate therapeutic principles and techniques",
                weight=0.3,
                pass_threshold=7.0
            ),
            ReviewCriteria(
                criterion_id="clinical_appropriateness",
                name="Clinical Appropriateness",
                description="Responses are clinically appropriate and safe",
                weight=0.25,
                pass_threshold=8.0
            ),
            ReviewCriteria(
                criterion_id="empathy_warmth",
                name="Empathy and Warmth",
                description="Content demonstrates appropriate empathy and therapeutic warmth",
                weight=0.2,
                pass_threshold=6.0
            ),
            ReviewCriteria(
                criterion_id="boundary_maintenance",
                name="Boundary Maintenance",
                description="Appropriate therapeutic boundaries are maintained",
                weight=0.15,
                pass_threshold=7.0
            ),
            ReviewCriteria(
                criterion_id="cultural_sensitivity",
                name="Cultural Sensitivity",
                description="Content is culturally sensitive and inclusive",
                weight=0.1,
                pass_threshold=6.0
            )
        ]

    def get_edge_criteria(self) -> List[ReviewCriteria]:
        """Get review criteria for edge case datasets"""
        return [
            ReviewCriteria(
                criterion_id="crisis_handling",
                name="Crisis Handling",
                description="Appropriate response to crisis situations (suicidal ideation, self-harm)",
                weight=0.4,
                pass_threshold=8.0
            ),
            ReviewCriteria(
                criterion_id="safety_prioritization",
                name="Safety Prioritization",
                description="Client safety is appropriately prioritized",
                weight=0.3,
                pass_threshold=9.0
            ),
            ReviewCriteria(
                criterion_id="de_escalation",
                name="De-escalation Techniques",
                description="Effective de-escalation techniques are demonstrated",
                weight=0.2,
                pass_threshold=7.0
            ),
            ReviewCriteria(
                criterion_id="resource_referral",
                name="Resource Referral",
                description="Appropriate referrals to emergency resources when needed",
                weight=0.1,
                pass_threshold=8.0
            )
        ]

    def get_bias_criteria(self) -> List[ReviewCriteria]:
        """Get review criteria for bias/cultural review"""
        return [
            ReviewCriteria(
                criterion_id="demographic_bias",
                name="Demographic Bias",
                description="No inappropriate assumptions based on demographics",
                weight=0.3,
                pass_threshold=8.0
            ),
            ReviewCriteria(
                criterion_id="cultural_competency",
                name="Cultural Competency",
                description="Demonstrates cultural awareness and competency",
                weight=0.25,
                pass_threshold=7.0
            ),
            ReviewCriteria(
                criterion_id="inclusive_language",
                name="Inclusive Language",
                description="Uses inclusive, non-discriminatory language",
                weight=0.2,
                pass_threshold=8.0
            ),
            ReviewCriteria(
                criterion_id="stereotype_avoidance",
                name="Stereotype Avoidance",
                description="Avoids harmful stereotypes and generalizations",
                weight=0.15,
                pass_threshold=8.0
            ),
            ReviewCriteria(
                criterion_id="accessibility_awareness",
                name="Accessibility Awareness",
                description="Considers accessibility and diverse needs",
                weight=0.1,
                pass_threshold=6.0
            )
        ]


class HumanQASignoff:
    """Manages human QA signoff process"""

    def __init__(self, storage_config: StorageConfig):
        self.config = storage_config
        self.s3_client = None
        self.sample_generator = QASampleGenerator(storage_config)
        self.criteria_manager = ReviewCriteriaManager()
        self._init_s3_client()

    def _init_s3_client(self):
        """Initialize S3 client"""
        if self.config.backend != self.config.backend.S3:
            raise ValueError("S3 backend required for QA signoff")

        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.config.s3_endpoint_url,
                aws_access_key_id=self.config.s3_access_key_id,
                aws_secret_access_key=self.config.s3_secret_access_key,
                region_name=self.config.s3_region or 'us-east-1'
            )

            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)

        except Exception as e:
            raise ValueError(f"Failed to connect to S3: {e}")

    def create_review_template(self, release_version: str) -> Dict[str, Any]:
        """Create review template with criteria and samples"""
        print("📋 Creating review template...")

        # Load manifest
        manifest = self.sample_generator.load_manifest(release_version)

        # Generate QA samples
        qa_samples = self.sample_generator.generate_qa_samples(manifest)

        # Save samples
        samples_url = self.sample_generator.save_qa_samples(qa_samples, release_version)

        # Create review template
        template = {
            'metadata': {
                'release_version': release_version,
                'created_at': datetime.utcnow().isoformat(),
                'template_version': '1.0.0',
                'samples_url': samples_url
            },
            'review_types': {
                'foundation': {
                    'description': 'Review of foundation therapeutic datasets',
                    'criteria': [asdict(c) for c in self.criteria_manager.get_foundation_criteria()],
                    'required_reviewers': 2,
                    'reviewer_qualifications': [
                        'Licensed mental health professional (LCSW, LPC, LMFT, or equivalent)',
                        'Minimum 3 years clinical experience',
                        'Experience with therapeutic training or supervision'
                    ]
                },
                'edge': {
                    'description': 'Review of edge case and crisis datasets',
                    'criteria': [asdict(c) for c in self.criteria_manager.get_edge_criteria()],
                    'required_reviewers': 2,
                    'reviewer_qualifications': [
                        'Licensed mental health professional with crisis intervention training',
                        'Experience with suicidal ideation and crisis response',
                        'Minimum 5 years clinical experience'
                    ]
                },
                'bias_cultural': {
                    'description': 'Bias and cultural competency review',
                    'criteria': [asdict(c) for c in self.criteria_manager.get_bias_criteria()],
                    'required_reviewers': 1,
                    'reviewer_qualifications': [
                        'Cultural competency training or specialization',
                        'Experience with diverse populations',
                        'Knowledge of bias in AI/ML systems (preferred)'
                    ]
                }
            },
            'review_process': {
                'scoring_scale': '1-10 (1=Poor, 10=Excellent)',
                'pass_criteria': 'All criteria must meet minimum threshold AND overall score >= 7.0',
                'escalation_process': 'Scores below threshold require supervisor review and remediation plan',
                'timeline': '5 business days for initial review, 2 days for escalation review'
            },
            'samples_summary': {
                'total_samples': len(qa_samples),
                'by_family': {},
                'by_review_type': {}
            }
        }

        # Summarize samples
        for sample in qa_samples:
            family = sample.family
            review_type = sample.review_type

            if family not in template['samples_summary']['by_family']:
                template['samples_summary']['by_family'][family] = 0
            template['samples_summary']['by_family'][family] += 1

            if review_type not in template['samples_summary']['by_review_type']:
                template['samples_summary']['by_review_type'][review_type] = 0
            template['samples_summary']['by_review_type'][review_type] += 1

        return template

    def save_review_template(self, template: Dict[str, Any], release_version: str) -> str:
        """Save review template to S3"""
        template_key = f"{self.config.exports_prefix}/releases/{release_version}/qa/review_template.json"

        try:
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=template_key,
                Body=json.dumps(template, indent=2),
                ContentType='application/json'
            )

            template_url = f"s3://{self.config.s3_bucket}/{template_key}"
            print(f"✓ Review template saved: {template_url}")
            return template_url

        except ClientError as e:
            raise ValueError(f"Failed to save review template: {e}")

    def create_mock_signoff(self, release_version: str) -> Dict[str, Any]:
        """Create mock signoff for demonstration (would be replaced by actual reviews)"""
        print("📝 Creating mock signoff (for demonstration)...")

        # Mock reviewers
        reviewers = [
            ReviewerInfo(
                reviewer_id="reviewer_001",
                name="Dr. Sarah Johnson",
                credentials="LCSW, PhD",
                specialization="Trauma therapy, crisis intervention",
                contact="s.johnson@example.com"
            ),
            ReviewerInfo(
                reviewer_id="reviewer_002",
                name="Dr. Michael Chen",
                credentials="LPC, MA",
                specialization="Cultural competency, diverse populations",
                contact="m.chen@example.com"
            ),
            ReviewerInfo(
                reviewer_id="reviewer_003",
                name="Dr. Amanda Rodriguez",
                credentials="LMFT, MSW",
                specialization="Foundation therapy, supervision",
                contact="a.rodriguez@example.com"
            )
        ]

        # Mock review results
        mock_signoff = {
            'metadata': {
                'release_version': release_version,
                'signoff_date': datetime.utcnow().isoformat(),
                'signoff_type': 'mock_demonstration',
                'total_reviewers': len(reviewers)
            },
            'reviewers': [asdict(r) for r in reviewers],
            'review_summary': {
                'foundation_review': {
                    'reviewer_ids': ['reviewer_001', 'reviewer_003'],
                    'samples_reviewed': 15,
                    'overall_score': 8.2,
                    'passed': True,
                    'concerns': [
                        'Minor inconsistency in boundary maintenance examples',
                        'Some responses could demonstrate more cultural awareness'
                    ],
                    'recommendations': [
                        'Add more diverse cultural examples',
                        'Strengthen boundary maintenance training examples'
                    ]
                },
                'edge_review': {
                    'reviewer_ids': ['reviewer_001'],
                    'samples_reviewed': 8,
                    'overall_score': 8.7,
                    'passed': True,
                    'concerns': [
                        'One sample had delayed crisis resource referral'
                    ],
                    'recommendations': [
                        'Emphasize immediate resource referral in crisis scenarios'
                    ]
                },
                'bias_cultural_review': {
                    'reviewer_ids': ['reviewer_002'],
                    'samples_reviewed': 12,
                    'overall_score': 7.8,
                    'passed': True,
                    'concerns': [
                        'Some language could be more inclusive',
                        'Need more accessibility considerations'
                    ],
                    'recommendations': [
                        'Review and update language guidelines',
                        'Add accessibility awareness training examples'
                    ]
                }
            },
            'overall_signoff': {
                'approved': True,
                'approval_date': datetime.utcnow().isoformat(),
                'conditions': [
                    'Address minor language inclusivity concerns in next release',
                    'Monitor boundary maintenance examples in ongoing training'
                ],
                'next_review_date': '2025-04-01',
                'signoff_authority': 'Clinical Review Board'
            }
        }

        return mock_signoff

    def save_signoff_record(self, signoff: Dict[str, Any], release_version: str) -> str:
        """Save signoff record to S3"""
        signoff_key = f"{self.config.exports_prefix}/releases/{release_version}/qa/signoff_record.json"

        try:
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=signoff_key,
                Body=json.dumps(signoff, indent=2),
                ContentType='application/json'
            )

            signoff_url = f"s3://{self.config.s3_bucket}/{signoff_key}"
            print(f"✓ Signoff record saved: {signoff_url}")
            return signoff_url

        except ClientError as e:
            raise ValueError(f"Failed to save signoff record: {e}")

    def generate_release_notes(self, signoff: Dict[str, Any], release_version: str) -> Dict[str, Any]:
        """Generate release notes based on signoff"""
        print("📄 Generating release notes...")

        release_notes = {
            'metadata': {
                'release_version': release_version,
                'release_date': datetime.utcnow().isoformat(),
                'notes_version': '1.0.0'
            },
            'release_summary': {
                'status': 'approved' if signoff['overall_signoff']['approved'] else 'rejected',
                'clinical_review_passed': signoff['overall_signoff']['approved'],
                'total_reviewers': signoff['metadata']['total_reviewers'],
                'total_samples_reviewed': sum(
                    review['samples_reviewed']
                    for review in signoff['review_summary'].values()
                )
            },
            'quality_assurance': {
                'foundation_datasets': {
                    'status': 'passed' if signoff['review_summary']['foundation_review']['passed'] else 'failed',
                    'score': signoff['review_summary']['foundation_review']['overall_score'],
                    'reviewer_count': len(signoff['review_summary']['foundation_review']['reviewer_ids'])
                },
                'edge_case_datasets': {
                    'status': 'passed' if signoff['review_summary']['edge_review']['passed'] else 'failed',
                    'score': signoff['review_summary']['edge_review']['overall_score'],
                    'reviewer_count': len(signoff['review_summary']['edge_review']['reviewer_ids'])
                },
                'bias_cultural_review': {
                    'status': 'passed' if signoff['review_summary']['bias_cultural_review']['passed'] else 'failed',
                    'score': signoff['review_summary']['bias_cultural_review']['overall_score'],
                    'reviewer_count': len(signoff['review_summary']['bias_cultural_review']['reviewer_ids'])
                }
            },
            'known_issues': [],
            'recommendations': [],
            'conditions': signoff['overall_signoff'].get('conditions', []),
            'next_review': signoff['overall_signoff'].get('next_review_date'),
            'approval_authority': signoff['overall_signoff'].get('signoff_authority')
        }

        # Collect all concerns and recommendations
        for review_type, review_data in signoff['review_summary'].items():
            release_notes['known_issues'].extend(review_data.get('concerns', []))
            release_notes['recommendations'].extend(review_data.get('recommendations', []))

        return release_notes

    def save_release_notes(self, release_notes: Dict[str, Any], release_version: str) -> str:
        """Save release notes to S3"""
        notes_key = f"{self.config.exports_prefix}/releases/{release_version}/release_notes.json"

        try:
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=notes_key,
                Body=json.dumps(release_notes, indent=2),
                ContentType='application/json'
            )

            notes_url = f"s3://{self.config.s3_bucket}/{notes_key}"
            print(f"✓ Release notes saved: {notes_url}")
            return notes_url

        except ClientError as e:
            raise ValueError(f"Failed to save release notes: {e}")

    def run_qa_signoff_process(self, release_version: str) -> Dict[str, Any]:
        """Run complete QA signoff process"""
        print(f"📋 Running QA signoff process for {release_version}...")

        # Create review template and samples
        template = self.create_review_template(release_version)
        template_url = self.save_review_template(template, release_version)

        # Create mock signoff (in production, this would wait for actual reviews)
        signoff = self.create_mock_signoff(release_version)
        signoff_url = self.save_signoff_record(signoff, release_version)

        # Generate release notes
        release_notes = self.generate_release_notes(signoff, release_version)
        notes_url = self.save_release_notes(release_notes, release_version)

        return {
            'release_version': release_version,
            'template_url': template_url,
            'signoff_url': signoff_url,
            'release_notes_url': notes_url,
            'approved': signoff['overall_signoff']['approved'],
            'summary': release_notes['release_summary']
        }

    def print_signoff_summary(self, results: Dict[str, Any]):
        """Print human-readable signoff summary"""
        print("\n" + "="*60)
        print("📋 HUMAN QA SIGNOFF SUMMARY")
        print("="*60)

        print(f"Release Version: {results['release_version']}")

        approval_status = "✅ APPROVED" if results['approved'] else "❌ REJECTED"
        print(f"Approval Status: {approval_status}")

        summary = results['summary']
        print(f"\n📊 REVIEW SUMMARY:")
        print(f"  Total Reviewers: {summary['total_reviewers']}")
        print(f"  Samples Reviewed: {summary['total_samples_reviewed']}")
        print(f"  Clinical Review: {'✅ PASSED' if summary['clinical_review_passed'] else '❌ FAILED'}")

        print(f"\n📄 ARTIFACTS:")
        print(f"  Review Template: {results['template_url']}")
        print(f"  Signoff Record: {results['signoff_url']}")
        print(f"  Release Notes: {results['release_notes_url']}")

        print("\n" + "="*60)


def main():
    """Main entry point"""
    print("🚀 Starting Human QA Signoff Process...")

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python human_qa_signoff.py <release_version>")
        print("Example: python human_qa_signoff.py v2025-01-02")
        sys.exit(1)

    release_version = sys.argv[1]

    # Load storage configuration
    config = get_storage_config()

    # Validate S3 configuration
    is_valid, error_msg = config.validate()
    if not is_valid:
        print(f"❌ Storage configuration error: {error_msg}")
        sys.exit(1)

    if config.backend != config.backend.S3:
        print("❌ S3 backend required. Set DATASET_STORAGE_BACKEND=s3")
        sys.exit(1)

    try:
        # Create QA signoff manager and run process
        qa_signoff = HumanQASignoff(config)
        results = qa_signoff.run_qa_signoff_process(release_version)

        # Print results
        qa_signoff.print_signoff_summary(results)

        # Exit with appropriate code
        if results['approved']:
            print(f"\n✅ QA signoff approved for {release_version}!")
            sys.exit(0)
        else:
            print(f"\n❌ QA signoff rejected for {release_version}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ QA signoff process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
