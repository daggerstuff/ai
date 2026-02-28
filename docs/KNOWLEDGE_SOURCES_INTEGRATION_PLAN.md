# Knowledge Sources Integration Plan

## Overview

This document outlines the integration of therapeutic books, PDFs, and reference materials from S3 into the Pixelated Empathy training and RAG systems.

## Current Status

- **Total Knowledge Sources**: 14 books/PDFs (27.25 MB)
- **Integration Status**: All 14 sources need migration to canonical `knowledge/` prefix
- **Priority Breakdown**:
  - Critical: 3 sources (DSM, Complex PTSD, IFS)
  - High: 6 sources (DBT, Brain Energy, PTSD resources)
  - Medium: 5 sources (Motivation, Social determinants, Self-help)

## Knowledge Sources Inventory

### Critical Priority (Must Integrate)

1. **Complex PTSD: From Surviving to Thriving** - Pete Walker
   - Current: `datasets/gdrive/raw/Books/Complex PTSD_...`
   - Target: `s3://pixel-data/knowledge/books/Complex_PTSD_From_Surviving_to_Thriving_Pete_Walker.epub`
   - Topics: complex_trauma, ptsd, recovery, emotional_flashbacks
   - Size: 683 KB

2. **Internal Family Systems Therapy** - Richard C. Schwartz
   - Current: `datasets/gdrive/raw/Books/Internal Family Systems...`
   - Target: `s3://pixel-data/knowledge/books/Internal_Family_Systems_Therapy_Richard_Schwartz.epub`
   - Topics: ifs_therapy, parts_work, self_leadership, trauma_therapy
   - Size: 450 KB

3. **DSM - Diagnostic and Statistical Manual**
   - Current: `datasets/gdrive/raw/Diagnostic and Statistical Manual...`
   - Target: `s3://pixel-data/knowledge/clinical/DSM_Diagnostic_and_Statistical_Manual.pdf`
   - Topics: diagnosis, clinical_criteria, mental_disorders
   - Size: 20 MB
   - **Note**: Core diagnostic reference - essential for clinical accuracy

### High Priority

1. **Brain Energy** - Christopher M. Palmer
   - Topics: neuroscience, mental_health, metabolism, brain_health
   - Size: 461 KB

2. **Loving Someone with PTSD** - Aphrodite T. Matsakis
   - Topics: ptsd, relationships, caregiver_support, trauma_recovery
   - Size: 406 KB

3. **The High Conflict Couple: A DBT Guide** - Alan E. Fruzzetti
   - Topics: dbt, couples_therapy, conflict_resolution, emotional_regulation
   - Size: 850 KB

4. **CBT and DBT For Anxiety** (EPUB + PDF)
   - Topics: cbt, dbt, anxiety, therapeutic_techniques
   - Size: 182 KB (EPUB) + 625 KB (PDF)

5. **Treating PTSD in Military Personnel: A Clinical Handbook**
   - Topics: ptsd, military, trauma, clinical_treatment
   - Size: 2.4 MB

### Medium Priority

1. **8 Keys to Eliminating Passive-Aggressiveness** - Andrea Brandt
2. **Addiction, Procrastination, and Laziness**
3. **Sedated: How Modern Capitalism Created our Mental Health Crisis**
4. **Psycho-Logical: Why Mental Health Goes Wrong**
5. **Self Help Psychology: Anxiety PTSD Recovery**

### Already Integrated (In Training Path)

- **The Gifts of Imperfection** - Brené Brown (86 KB)
  - Path: `s3://pixel-data/training/v1/stage1_foundation/knowledge/`
- **The Myth of Normal** - Gabor Maté (184 KB)
  - Path: `s3://pixel-data/training/v1/stage1_foundation/knowledge/`

## Integration Pipeline

### Step 1: Migrate to Knowledge Prefix ✅ READY

**Script**: `ai/scripts/migrate_knowledge_sources.py`

```bash
# Run migration (currently commented out for safety)
uv run python ai/scripts/migrate_knowledge_sources.py
```

This will:

- Copy all books/PDFs from current S3 locations to `s3://pixel-data/knowledge/`
- Organize by category: `books/`, `clinical/`
- Preserve original files (uses `rclone copy`, not `move`)

### Step 2: Extract Text Content

**Tools Required**:

- `calibre` (for EPUB/AZW3 → text)
- `pdftotext` (for PDF → text)

**Process**:

1. Download source file from S3
2. Convert to plain text using appropriate tool
3. Upload `.txt` version alongside original
4. Store in same directory structure

**Example**:

```bash
# EPUB extraction
ebook-convert Complex_PTSD.epub Complex_PTSD.txt --enable-heuristics

# PDF extraction
pdftotext -layout DSM_Manual.pdf DSM_Manual.txt
```

### Step 3: Chunk and Embed

**Configuration**:

- Chunk size: 500 tokens
- Overlap: 50 tokens
- Embedding model: `all-MiniLM-L6-v2` (same as YouTubeRAGSystem)

**Output**: Embeddings stored for RAG retrieval

### Step 4: Integrate with RAG System

**Target**: `ai/pipelines/youtube_rag_system.py`

**Modifications Needed**:

1. Add `knowledge_sources_dir` parameter to `YouTubeRAGSystem`
2. Extend `build_rag_index()` to process book chunks
3. Add metadata for source attribution (book title, author, page/chapter)

**Example Integration**:

```python
class YouTubeRAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", include_books: bool = True):
        # ... existing code ...
        self.knowledge_sources_dir = Path("knowledge/books")
        if include_books:
            self._load_knowledge_sources()
    
    def _load_knowledge_sources(self):
        """Load and index therapeutic books."""
        for book_file in self.knowledge_sources_dir.glob("*.txt"):
            # Process book content
            # Add to RAG index with metadata
```

### Step 5: Augment Training Data

**Method**: Use book content for few-shot examples and knowledge grounding

**Applications**:

1. **Few-Shot Examples**: Extract therapeutic dialogues from books
2. **Knowledge Grounding**: Add factual context to generated responses
3. **Persona Development**: Use author voice/style for persona training

## Environment Variables

Add to `.env.example`:

```bash
# Knowledge Sources
KNOWLEDGE_SOURCES_ENABLED=true
KNOWLEDGE_SOURCES_PATH=s3://pixel-data/knowledge
EXTRACT_BOOK_TEXT=true
```

## Next Steps

1. **Review and Approve** this integration plan
2. **Uncomment migration code** in `migrate_knowledge_sources.py`
3. **Run migration** to move files to canonical locations
4. **Install extraction tools**: `calibre`, `pdftotext`
5. **Extract text** from all sources
6. **Update RAG system** to include book content
7. **Test retrieval** with sample queries
8. **Monitor quality** of book-augmented responses

## Quality Assurance

### Verification Checklist

- [ ] All 14 sources migrated to `knowledge/` prefix
- [ ] Text extracted from all EPUBs and PDFs
- [ ] Embeddings generated for all chunks
- [ ] RAG system returns relevant book excerpts
- [ ] Source attribution included in responses
- [ ] No duplicate content between books and training data

### Testing Queries

1. "What are the characteristics of complex PTSD according to Pete Walker?"
2. "How does Internal Family Systems therapy work?"
3. "What are the DSM criteria for PTSD?"
4. "How can DBT help with high-conflict relationships?"

## Benefits

1. **Enhanced Knowledge Base**: 27+ MB of therapeutic expertise
2. **Clinical Accuracy**: DSM and clinical handbooks for diagnostic precision
3. **Diverse Perspectives**: Multiple therapeutic modalities (IFS, DBT, CBT)
4. **Evidence-Based**: Peer-reviewed and clinically validated content
5. **RAG Enhancement**: Richer context for conversational responses

## Registry Files

- **Knowledge Sources**: `ai/data/knowledge_sources_registry.json`
- **Migration Script**: `ai/scripts/migrate_knowledge_sources.py`
- **Dataset Registry**: `ai/data/dataset_registry.json` (for training data)

---

**Status**: ✅ Ready for execution
**Last Updated**: 2026-02-03
**Owner**: AI Training Pipeline
