# Therapy Dataset Sourcing - User Guide

**Specialized module for finding high-quality therapy conversation datasets**

---

## 🎯 Overview

The Therapy Dataset Sourcing module is designed to find and filter therapy/counseling conversation datasets from various sources (primarily HuggingFace Hub) with specific criteria:

✅ **Multi-turn conversations** (20+ turns preferred)  
✅ **Therapy/mental health focus**  
✅ **Quality scoring** and validation  
✅ **Automatic filtering** and ranking  
✅ **Comprehensive metadata** extraction

---

## 🚀 Quick Start

### Basic Usage

```python
from ai.academic_sourcing import find_therapy_datasets

# Find datasets with 20+ turn conversations
datasets = find_therapy_datasets(
    min_turns=20,
    min_quality=0.5
)

# Results are automatically saved to:
# ai/training_ready/datasets/sourced/therapy_datasets.json
```

### Advanced Usage

```python
from ai.academic_sourcing import TherapyDatasetSourcing

# Create sourcing engine
sourcing = TherapyDatasetSourcing()

# Search HuggingFace Hub
datasets = sourcing.search_huggingface(
    query="therapy conversation mental health",
    min_turns=20,
    limit=50
)

# Apply filters
datasets = sourcing.filter_by_quality(datasets, min_quality=0.6)
datasets = sourcing.filter_by_therapeutic_relevance(datasets, min_relevance=0.6)

# Rank by composite score
datasets = sourcing.rank_datasets(datasets)

# Export results
sourcing.export_results(datasets, "my_datasets.json")

# Generate report
report = sourcing.generate_report(datasets)
print(report)
```

---

## 📊 Dataset Metadata

Each dataset includes comprehensive metadata:

```python
@dataclass
class DatasetMetadata:
    name: str                          # Dataset ID (e.g., "user/dataset-name")
    source: str                        # Source type ("huggingface", "github", etc.)
    url: str                           # Direct URL to dataset
    description: str                   # Dataset description
    tags: List[str]                    # Tags/keywords
    downloads: int                     # Number of downloads
    likes: int                         # Number of likes/stars
    size_bytes: Optional[int]          # Dataset size in bytes
    num_conversations: Optional[int]   # Number of conversations
    avg_turns: Optional[float]         # Average turns per conversation
    min_turns: Optional[int]           # Minimum turns
    max_turns: Optional[int]           # Maximum turns
    conversation_format: Optional[str] # "multi_turn", "medium_turn", etc.
    languages: List[str]               # Supported languages
    license: Optional[str]             # Dataset license
    created_at: Optional[str]          # Creation date
    updated_at: Optional[str]          # Last update date
    quality_score: float               # Quality score (0.0-1.0)
    therapeutic_relevance: float       # Therapeutic relevance (0.0-1.0)
```

---

## 🔍 Search & Filtering

### 1. Search HuggingFace Hub

```python
datasets = sourcing.search_huggingface(
    query="therapy counseling dialogue",
    min_turns=15,  # Minimum conversation turns
    limit=30       # Maximum results
)
```

**Search Tips:**

- Use specific therapy terms: "CBT", "DBT", "trauma therapy"
- Combine with "conversation", "dialogue", "session"
- Try variations: "counseling", "psychotherapy", "mental health"

### 2. Filter by Conversation Length

```python
# Only datasets with 20+ turn conversations
long_conversations = sourcing.filter_by_conversation_length(
    datasets,
    min_turns=20,
    max_turns=None  # No maximum
)

# Medium-length conversations (10-19 turns)
medium_conversations = sourcing.filter_by_conversation_length(
    datasets,
    min_turns=10,
    max_turns=19
)
```

### 3. Filter by Quality

```python
# High-quality datasets only
high_quality = sourcing.filter_by_quality(
    datasets,
    min_quality=0.7  # 0.0-1.0 scale
)
```

**Quality Score Factors:**

- Downloads (max 0.3)
- Likes/stars (max 0.2)
- Quality indicators in description (max 0.3)
- Has license (0.1)
- Recent updates (0.1)

### 4. Filter by Therapeutic Relevance

```python
# Highly relevant to therapy/mental health
relevant = sourcing.filter_by_therapeutic_relevance(
    datasets,
    min_relevance=0.6  # 0.0-1.0 scale
)
```

**Relevance Score Based On:**

- Therapy keywords in name/description/tags
- Mental health terminology
- Clinical/therapeutic focus

---

## 📈 Ranking & Scoring

### Default Ranking

```python
# Rank by composite score (balanced)
ranked = sourcing.rank_datasets(datasets)
```

**Default Weights:**

- Quality: 30%
- Therapeutic Relevance: 30%
- Conversation Length: 20%
- Popularity (downloads): 20%

### Custom Ranking

```python
# Prioritize long conversations
custom_weights = {
    "quality": 0.2,
    "relevance": 0.2,
    "turns": 0.5,      # 50% weight on conversation length
    "popularity": 0.1
}

ranked = sourcing.rank_datasets(datasets, weights=custom_weights)
```

**Use Cases:**

- **Research**: Prioritize quality + relevance
- **Training Data**: Prioritize conversation length + quality
- **Popular Datasets**: Prioritize downloads + likes

---

## 💾 Export & Reporting

### Export to JSON

```python
# Export filtered/ranked datasets
output_path = sourcing.export_results(
    datasets,
    filename="therapy_datasets_20plus_turns.json"
)

print(f"Saved to: {output_path}")
```

**Output Format:**

```json
[
  {
    "name": "user/therapy-conversations",
    "source": "huggingface",
    "url": "https://huggingface.co/datasets/user/therapy-conversations",
    "avg_turns": 25.3,
    "quality_score": 0.85,
    "therapeutic_relevance": 0.92,
    "downloads": 15420,
    ...
  }
]
```

### Generate Report

```python
report = sourcing.generate_report(datasets)
print(report)
```

**Report Includes:**

- Total datasets found
- Average conversation turns
- Quality distribution
- Top 10 datasets with details

---

## 🎯 Use Cases

### 1. Find Long Therapy Conversations

```python
# Find datasets with 20+ turn conversations
datasets = find_therapy_datasets(
    min_turns=20,
    min_quality=0.5
)
```

### 2. Find High-Quality CBT Datasets

```python
sourcing = TherapyDatasetSourcing()

# Search for CBT-specific datasets
datasets = sourcing.search_huggingface(
    query="cognitive behavioral therapy CBT dialogue",
    min_turns=15,
    limit=30
)

# Filter for high quality
datasets = sourcing.filter_by_quality(datasets, min_quality=0.7)
datasets = sourcing.rank_datasets(datasets)
```

### 3. Find Diverse Therapy Datasets

```python
# Search multiple queries
queries = [
    "therapy conversation",
    "counseling dialogue",
    "mental health session",
    "psychotherapy transcript"
]

all_datasets = []
for query in queries:
    results = sourcing.search_huggingface(query, min_turns=10, limit=20)
    all_datasets.extend(results)

# Deduplicate and rank
unique_datasets = {d.name: d for d in all_datasets}.values()
ranked = sourcing.rank_datasets(list(unique_datasets))
```

### 4. Custom Quality Criteria

```python
# Find datasets with specific criteria
datasets = sourcing.search_huggingface("therapy", limit=50)

# Custom filtering
filtered = [
    d for d in datasets
    if d.avg_turns and d.avg_turns >= 25  # Very long conversations
    and d.downloads >= 1000                # Popular
    and d.license                          # Has license
    and "2024" in (d.updated_at or "")     # Recently updated
]

# Rank by conversation length
custom_weights = {"turns": 0.6, "quality": 0.2, "relevance": 0.2}
ranked = sourcing.rank_datasets(filtered, weights=custom_weights)
```

---

## 🔑 Configuration

### Environment Variables

```bash
# HuggingFace token (optional but recommended for higher rate limits)
export HUGGINGFACE_TOKEN="your-token-here"

# GitHub token (for future GitHub dataset search)
export GITHUB_TOKEN="your-token-here"
```

**Get HuggingFace Token:**

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Set the environment variable

---

## 📋 Conversation Format Classification

Datasets are automatically classified by conversation length:

| Format          | Turns | Use Case                                    |
| --------------- | ----- | ------------------------------------------- |
| **MULTI_TURN**  | 20+   | Extended therapy sessions, deep exploration |
| **MEDIUM_TURN** | 10-19 | Standard therapy conversations              |
| **SHORT_TURN**  | 5-9   | Brief interventions, check-ins              |
| **SINGLE_TURN** | 1-4   | Q&A, single exchanges                       |

---

## 🎨 Example Workflow

```python
from ai.academic_sourcing import TherapyDatasetSourcing

# 1. Initialize
sourcing = TherapyDatasetSourcing()

# 2. Search multiple sources
datasets = []
for query in ["therapy conversation", "counseling dialogue", "mental health session"]:
    results = sourcing.search_huggingface(query, min_turns=20, limit=20)
    datasets.extend(results)

# 3. Filter
datasets = sourcing.filter_by_conversation_length(datasets, min_turns=20)
datasets = sourcing.filter_by_quality(datasets, min_quality=0.6)
datasets = sourcing.filter_by_therapeutic_relevance(datasets, min_relevance=0.6)

# 4. Rank
datasets = sourcing.rank_datasets(datasets)

# 5. Export top 10
top_datasets = datasets[:10]
sourcing.export_results(top_datasets, "top_therapy_datasets.json")

# 6. Generate report
print(sourcing.generate_report(top_datasets))
```

---

## 🚀 Running the Demo

```bash
cd /home/vivi/pixelated
uv run python ai/academic_sourcing/demo_therapy_sourcing.py
```

**Demo Includes:**

1. Basic search for therapy datasets
2. Filtered search (high quality + high relevance)
3. Full pipeline with report generation
4. Custom ranking (prioritizing long conversations)

---

## 📈 Performance Tips

1. **Use Specific Queries**: More specific = better results
   - ❌ "therapy"
   - ✅ "cognitive behavioral therapy conversation"

2. **Adjust Filters**: Balance quantity vs quality
   - Strict filters (min_quality=0.7): Fewer, higher quality
   - Relaxed filters (min_quality=0.4): More options, manual review needed

3. **Custom Weights**: Optimize for your use case
   - Training data: Prioritize `turns` and `quality`
   - Research: Prioritize `relevance` and `quality`
   - Popular datasets: Prioritize `popularity`

4. **Batch Processing**: Search multiple queries and combine results

---

## 🔧 Troubleshooting

### "No datasets found"

- Try broader search terms
- Lower `min_turns` threshold
- Check HuggingFace Hub manually for available datasets

### "Rate limit exceeded"

- Set `HUGGINGFACE_TOKEN` environment variable
- Add delays between searches
- Reduce `limit` parameter

### "Cannot determine conversation statistics"

- Some datasets don't expose sample data
- Manual inspection may be needed
- Check dataset card on HuggingFace Hub

---

## 📚 Related Documentation

- [Academic Sourcing README](./README.md) - Main module documentation
- [Enhancement Summary](./ENHANCEMENT_SUMMARY.md) - Recent feature additions
- [HuggingFace Datasets](https://huggingface.co/datasets) - Browse available datasets

---

**Happy dataset hunting!** 🎯

For questions or issues, check the demo script or module source code for examples.
