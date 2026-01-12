# Academic Sourcing Module

> **Comprehensive academic literature and dataset sourcing for psychology and therapy research**

**Status**: ✅ Production Ready | **Version**: 2.0 | **Last Updated**: 2026-01-12

---

## 🎯 Quick Start

```python
from ai.academic_sourcing import AcademicSourcingEngine

# Search across 20+ sources (12 publishers + 8 APIs)
engine = AcademicSourcingEngine()
results = engine.search_literature("trauma therapy", limit=50)

# Export results
engine.export_results(results, "trauma_therapy_books.json")
```

---

## 📚 What's Included

### Literature Sourcing (20+ Sources)

**Publishers (12):**

- APA, Springer Nature, Oxford, Cambridge, Wiley, Elsevier
- Taylor & Francis, SAGE, Guilford, Routledge, JSTOR

**APIs (8):**

- ArXiv, Semantic Scholar, CrossRef, PubMed
- OpenAlex, CORE, Europe PMC, Google Scholar

### Specialized Tools

**Therapy Dataset Sourcing:**

- HuggingFace Hub search
- 20+ turn conversation filtering
- Quality & relevance scoring
- See: [THERAPY_DATASET_GUIDE.md](./THERAPY_DATASET_GUIDE.md)

**DOI Resolution:**

- CrossRef & DataCite APIs
- Batch resolution
- Psychology-specific search

---

## 🚀 Installation & Setup

### Environment Variables

```bash
# Publishers (optional but recommended)
export APA_API_KEY="your-key"
export SPRINGER_API_KEY="your-key"
export OXFORD_API_KEY="your-key"
export CAMBRIDGE_API_KEY="your-key"
export WILEY_API_KEY="your-key"
export ELSEVIER_API_KEY="your-key"

# APIs (optional - improves rate limits)
export SEMANTIC_SCHOLAR_API_KEY="your-key"
export PUBMED_API_KEY="your-key"
export CORE_API_KEY="your-key"
export HUGGINGFACE_TOKEN="your-token"
```

**Note**: Most sources work without API keys, but keys improve rate limits and access.

### Frontend Configuration

The frontend requires the backend API URL to be configured.

```bash
# .env (Development)
PUBLIC_ACADEMIC_API_URL="http://localhost:8000/api"

# .env (Production)
PUBLIC_ACADEMIC_API_URL="https://your-production-api.com/api"
```

---

## 💡 Usage Examples

### Basic Literature Search

```python
from ai.academic_sourcing import AcademicSourcingEngine

engine = AcademicSourcingEngine()

# Search all sources
results = engine.search_literature("cognitive behavioral therapy", limit=30)

# Results include books from all 12 publishers + 8 APIs
for book in results:
    print(f"{book.title} by {', '.join(book.authors)}")
    print(f"  Relevance: {book.therapeutic_relevance_score:.2f}")
    print(f"  Publisher: {book.publisher}")
```

### Strategy-Based Searching

```python
from ai.academic_sourcing import SourcingStrategy

# API sources only (fast)
engine = AcademicSourcingEngine(strategy=SourcingStrategy.API_ONLY)
api_results = engine.search_literature("anxiety disorders", limit=20)

# Publishers only (high quality)
engine = AcademicSourcingEngine(strategy=SourcingStrategy.PUBLISHER_ONLY)
publisher_results = engine.search_literature("depression treatment", limit=20)

# Hybrid (recommended - comprehensive)
engine = AcademicSourcingEngine(strategy=SourcingStrategy.HYBRID)
all_results = engine.search_literature("PTSD therapy", limit=50)
```

### Specific Source Search

```python
from ai.academic_sourcing import SourceType

# Search specific publisher
oxford_books = engine.fetch_from_publisher(
    SourceType.OXFORD,
    "trauma therapy",
    limit=15
)

# Search specific API
pubmed_papers = engine.fetch_pubmed("DBT therapy", limit=20)
```

### Therapy Dataset Discovery

```python
from ai.academic_sourcing import find_therapy_datasets

# Find datasets with 20+ turn conversations
datasets = find_therapy_datasets(
    min_turns=20,
    min_quality=0.6
)

# Results saved to: ai/training_ready/datasets/sourced/therapy_datasets.json
```

### DOI Resolution

```python
from ai.academic_sourcing.doi_resolution import DOIResolver

resolver = DOIResolver()

# Resolve single DOI
metadata = resolver.resolve("10.1037/a0012345")
print(f"Title: {metadata.title}")
print(f"Authors: {', '.join(metadata.authors)}")

# Batch resolution
dois = ["10.1037/a0012345", "10.1016/j.cpr.2020.101832"]
results = resolver.batch_resolve(dois)
```

---

## 📊 Features

### Automatic Quality Assessment

All results include:

- **Therapeutic Relevance Score** (0.0-1.0)
- **Stage Assignment** (foundation vs therapeutic expertise)
- **Automatic Deduplication**
- **Metadata Extraction** (title, authors, abstract, keywords, etc.)

### Export Formats

```python
# Export to JSON
engine.export_results(results, "my_results.json")

# Access metadata
for book in results:
    data = book.to_dict()  # Convert to dictionary
    # Use data for further processing
```

---

## 🏗️ Architecture

### Module Structure

```
ai/academic_sourcing/
├── academic_sourcing.py          # Main engine
├── therapy_dataset_sourcing.py   # HuggingFace dataset search
├── publishers/                   # 12 publisher integrations
├── doi_resolution/              # DOI resolution
├── tests/                       # Test suite
└── docs/                        # Documentation
```

### Sourcing Strategies

1. **API_ONLY**: Fast, broad coverage (8 APIs)
2. **PUBLISHER_ONLY**: High quality, authenticated (12 publishers)
3. **SCRAPING_ONLY**: Web scraping fallback
4. **HYBRID**: All sources (recommended)

---

## 🖥️ Frontend Interface

The module now includes a modern, responsive web interface for searching and research.

### Literature Search (`/research`)

- **Unified Search**: Query across all enabled sources simultaneously.
- **Advanced Filters**:
  - Year range (1900-Present)
  - Therapeutic topics (CBT, DBT, Trauma, etc.)
  - Relevance threshold sliders
- **Export**: Download results in JSON, CSV, BibTeX, or RIS formats.
- **Detailed Cards**: View metadata, relevance scores, and direct links.

### Dataset Discovery (`/research/datasets`)

- **Therapeutic Focus**: Specialized search for therapy conversation datasets.
- **Quality Metrics**: Filter by conversation quality, turn count, and therapeutic relevance.
- **Direct Access**: Preview and link directly to HuggingFace datasets.

---

## 🎓 Publisher Coverage

| Publisher            | Specialty               | API Required |
| -------------------- | ----------------------- | ------------ |
| **APA**              | Clinical Psychology     | Yes          |
| **Springer**         | General Psychology      | Yes          |
| **Oxford**           | Clinical/Neuroscience   | Yes          |
| **Cambridge**        | Research/Developmental  | Yes          |
| **Wiley**            | Clinical Practice       | Yes          |
| **Elsevier**         | Neuroscience/Biological | Yes          |
| **Taylor & Francis** | General                 | Yes          |
| **SAGE**             | Social Science          | Yes          |
| **Guilford**         | Evidence-Based Therapy  | Optional     |
| **Routledge**        | Psychotherapy           | Yes          |
| **JSTOR**            | Academic Archives       | Yes          |

---

## 📈 API Sources

| Source               | Coverage          | Specialty           |
| -------------------- | ----------------- | ------------------- |
| **ArXiv**            | Physics/CS/Psych  | Open access         |
| **Semantic Scholar** | AI-powered search | General             |
| **CrossRef**         | DOI resolution    | Metadata            |
| **PubMed**           | 35M+ citations    | Biomedical          |
| **OpenAlex**         | 250M+ works       | Open bibliographic  |
| **CORE**             | 200M+ papers      | Open access         |
| **Europe PMC**       | 40M+ records      | European biomedical |
| **Google Scholar**   | Comprehensive     | Web scraping        |

---

## 🧪 Testing

Run the test suite:

```bash
cd /home/vivi/pixelated
uv run python -m pytest ai/academic_sourcing/tests/ -v
```

---

## 📖 Additional Documentation

- **[THERAPY_DATASET_GUIDE.md](./THERAPY_DATASET_GUIDE.md)** - Comprehensive guide for therapy dataset sourcing
- **Demo Script**: `demo_therapy_sourcing.py` - Working examples

---

## 🔧 Troubleshooting

### "No results found"

- Check API keys are set
- Try broader search terms
- Use HYBRID strategy for maximum coverage

### "Authentication failed"

- Verify API keys are correct
- Check institutional access for publishers
- Some publishers require paid subscriptions

### "Rate limit exceeded"

- Set API keys to increase limits
- Add delays between searches
- Reduce limit parameter

---

## 📝 Development History

**Version 2.0** (2026-01-12):

- ✅ Consolidated 3 separate implementations
- ✅ Added 4 new API sources (PubMed, OpenAlex, CORE, Europe PMC)
- ✅ Implemented 11 new publishers (total: 12)
- ✅ Added therapy dataset sourcing
- ✅ Added DOI resolution
- ✅ Complete test suite
- ✅ Production ready

**Version 1.0** (Previous):

- Basic ArXiv + Semantic Scholar integration
- APA publisher integration
- Google Scholar scraping

---

## 🎯 Use Cases

### Research

- Literature reviews
- Citation discovery
- Topic exploration

### Training Data

- Dataset discovery
- Conversation sourcing
- Quality filtering

### Clinical Practice

- Evidence-based resources
- Treatment manuals
- Assessment tools

---

## 🤝 Contributing

The module is designed to be extensible:

1. **Add Publishers**: Extend `BasePublisher` class
2. **Add APIs**: Add methods to `AcademicSourcingEngine`
3. **Add Tests**: Add to `tests/` directory

---

## 📄 License

Part of the Pixelated Empathy project.

---

## 🆘 Support

For issues or questions:

1. Check this README
2. Review [THERAPY_DATASET_GUIDE.md](./THERAPY_DATASET_GUIDE.md)
3. Check demo scripts
4. Review test files for examples

---

**Built with ❤️ for psychology and therapy research**
