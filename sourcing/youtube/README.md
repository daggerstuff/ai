# YouTube Channel Curation System

**Purpose:** Discover, evaluate, and curate therapeutic YouTube channels for
CPTSD dataset training. **Agent:** PIX-28 - YouTube Source Expansion
**Timeline:** Week 1-2 (3-5 days target)

## Quick Start

### Installation

```bash
# Install dependencies
pip install pydantic

# YouTube Data API key required
export YOUTUBE_API_KEY="your-api-key-here"
```

### Run Discovery

```bash
# Python CLI
python -m ai.sourcing.youtube.cli discover \
    --api-key $YOUTUBE_API_KEY \
    --channels 50 \
    --output qualified_channels.json \
    --report discovery_report.md

# Or programmatically
from ai.sourcing.youtube.processor import run_pipeline

results, report = run_pipeline(
    api_key="your-api-key",
    target_channels=50,
    output_path="qualified_channels.json"
)

print(f"Found {len(results.qualified_channels)} channels")
```

### Check Channel Health

```bash
python -m ai.sourcing.youtube.cli check \
    --channel-id UCxxxxxxxxxxxxxx
```

### List Registry

```bash
python -m ai.sourcing.youtube.cli list
```

## Architecture

```bash
ai/sourcing/youtube/
├── __init__.py           # Module exports
├── models.py             # Data models (Channel, QualityMetrics, etc.)
├── api.py                # YouTube API integration (ChannelHunter, Analyzer)
├── monitoring.py         # Health checks and monitoring (ChannelMonitor)
├── processor.py          # Main pipeline orchestrator
└── cli.py                # Command-line interface
```

## Quality Scoring (0.0-1.0)

| Metric             | Weight | Description                        |
| ------------------ | ------ | ---------------------------------- |
| Clinical Accuracy  | 30%    | Evidence-based therapeutic content |
| Content Quality    | 25%    | Educational value and clarity      |
| Production Quality | 15%    | Audio/video technical quality      |
| Credibility        | 15%    | Professional credentials           |
| Engagement Quality | 10%    | Interaction and community          |
| Consistency        | 5%     | Posting regularity                 |

## Acceptance Criteria

Channel must meet ALL:

- Quality score ≥ 0.8
- Clinical accuracy ≥ 0.80 (highest weight)
- Production quality ≥ 0.70
- Credibility score ≥ 0.75
- Minimum 1,000 subscribers
- Minimum 20 videos
- Professional credential OR reputable organization

## Categories

10 therapeutic categories:

1. CPTSD Education
2. Trauma-Informed Care
3. DBT Skills
4. CBT Techniques
5. Somatic Therapy
6. EMDR Therapy
7. Mindfulness
8. Crisis Support
9. Professional Training
10. Recovery/Patient Stories

## Keywords by Category

Each category has specific search terms for discovery:

**CPTSD:**

- cptsd, complex ptsd, trauma symptoms, developmental trauma
- childhood trauma, narcissistic abuse, c-ptsd recovery

**DBT Skills:**

- dbt, distress tolerance, emotion regulation, wise mind
- opposite action, radical acceptance, dialectical behavior therapy

**Mindfulness:**

- mindfulness, guided meditation, grounding techniques, 5-4-3-2-1
- body scan, breathwork, meditation for trauma

**Crisis Support:**

- crisis support, suicide prevention, safety plan, emergency mental health
- crisis hotline, crisis intervention, help resources

And 6 more categories defined in `api.py`.

## Alerts

Monitoring system has 4 alert levels:

| Severity | Trigger                   | Action                |
| -------- | ------------------------- | --------------------- |
| CRITICAL | Channel removed           | Immediate replacement |
| ERROR    | Subscribers drop >10%     | Investigate cause     |
| WARNING  | No new content 30+ days   | Monitor 14 more days  |
| INFO     | Quality dropped below 0.7 | Schedule review       |

## Configuration

Edit `ai/sourcing/youtube/api.py` to customize:

```python
from ai.sourcing.youtube.models import ChannelHunterConfig

config = ChannelHunterConfig(
    min_subscribers=1000,  # Minimum 1K subscribers
    min_videos=20,         # At least 20 videos
    target_channels=50,    # Find 50 channels total
    target_languages={"en", "es", "fr", "de"},  # Supported languages
    require_professional=True,  # Must be professional source
    quality_threshold=0.8,  # 80% minimum quality
)
```

## Output Files

- **qualified_channels.json** - All qualified channels with full metadata
- **discovery_report.md** - Comprehensive markdown report
- **health_report.txt** - Channel health monitoring report

## Sample Channel Data

```json
{
  "channel_id": "UC...",
  "channel_name": "Therapist for Trauma Recovery",
  "channel_url": "https://www.youtube.com/@traumatherapist",
  "subscriber_count": 150000,
  "video_count": 500,
  "languages": ["en"],
  "categories": ["trauma_informed", "cptsd_education"],
  "quality_score": 0.85,
  "is_professional": true,
  "credentials": ["licensed clinical social worker", "phd psychology"],
  "licensing": {
    "cc_license": true,
    "commercial_use": false,
    "attribution_required": true
  }
}
```

## Dependencies

- `pydantic` - Data validation
- `google-api-python-client` - YouTube API (optional, can use requests)
- `yt-dlp` - For video metadata extraction
- `whisper` / `faster-whisper` - For audio quality assessment (TODO)

## Known Limitations

1. **YouTube API Limits** - Quota management required (recommend multiple API
   keys)
2. **Licensing Verification** - Automated detection limited; manual review
   recommended
3. **Channel Status** - Initial status assumed ACTIVE; monitoring system updates
   over time
4. **Video Quality Metrics** - Requires audio download and analysis (performance
   consideration)

## Next Steps After Discovery

1. **PIX-30 (Pipeline Architect):** Use qualified channels for test data
2. **PIX-31 (Multilingual Specialist):** Extend pipeline for 15+ languages
3. **PIX-29 (Dedup Engineer):** Use transcriptions for deduplication

## See Also

- `/ai/YouTube_Transcription_Pipeline.ipynb` - Audio transcription pipeline
- `.agent/internal/plans/phase-1-task-distribution.md` - Wave 1 task breakdown
- `.agent/internal/plans/wave-detailed-review.md` - Detailed wave analysis

## Contact

For issues or questions, reference the CPTSD epic (PIX-95) in Linear.
