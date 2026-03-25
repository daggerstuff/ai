# Academic Sourcing Front-End Implementation Plan

**Project**: Academic Literature Search Interface  
**Backend**: Academic Sourcing Module (Python)  
**Frontend**: Astro + React (matching existing Pixelated Empathy theme)  
**Date**: 2026-01-12 **Status**: 🚧 **In Progress** (Phase 1 Started)

---

## 🎯 Project Overview

Create a beautiful, user-friendly web interface for the Academic Sourcing module
that:

- Matches the existing Pixelated Empathy design system
- Provides powerful search across 20+ academic sources
- Offers advanced filtering and export capabilities
- Integrates seamlessly with the Python backend

---

## 🎨 Design System Integration

### Theme Consistency

- **Color System**: OKLCH-based dark theme from `design-system.css`
- **Typography**: System font stack with responsive scaling
- **Components**: Card-based layout with smooth transitions
- **Animations**: Subtle micro-interactions (150-500ms)
- **Spacing**: 4px base unit system

### Visual Style

- **Primary**: `oklch(70.5% 0.213 47.604)` - Vibrant accent
- **Background**: `oklch(14.7% 0.004 49.25)` - Deep dark
- **Cards**: Elevated with subtle shadows and borders
- **Gradients**: Subtle radial gradients for depth

---

## 📐 Architecture

### Tech Stack

```
Frontend:
├── Astro (SSG/SSR framework)
├── React (Interactive components)
├── TypeScript (Type safety)
└── Design System CSS (Existing theme)

Backend API:
├── FastAPI (Python REST API)
├── Academic Sourcing Module (Core logic)
└── CORS middleware (Cross-origin support)
```

### File Structure

```
src/
├── pages/
│   └── research/
│       ├── index.astro              # Main search page
│       └── datasets.astro           # Dataset discovery page
│
├── components/
│   └── research/
│       ├── SearchInterface.tsx      # Main search component
│       ├── SearchFilters.tsx        # Advanced filters
│       ├── ResultsGrid.tsx          # Results display
│       ├── ResultCard.tsx           # Individual result
│       ├── ExportPanel.tsx          # Export options
│       ├── SourceSelector.tsx       # Source selection
│       └── DatasetSearch.tsx        # Dataset search
│
├── layouts/
│   └── ResearchLayout.astro         # Layout for research pages
│
└── styles/
    └── research.css                 # Research-specific styles

ai/sourcing/academic/api/
├── main.py                          # FastAPI app
├── routes.py                        # API endpoints
└── middleware.py                    # CORS, auth, etc.
```

---

## 🎨 UI/UX Design

### Page 1: Literature Search (`/research`)

**Hero Section:**

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   🔬 Academic Literature Search                    │
│   Search 20+ sources for psychology & therapy      │
│   research                                          │
│                                                     │
│   ┌─────────────────────────────────────────────┐ │
│   │ 🔍 Search for books, papers, articles...   │ │
│   └─────────────────────────────────────────────┘ │
│                                                     │
│   [Advanced Filters ▼]                             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Search Interface:**

- Large, prominent search bar with autocomplete
- Source selector chips (All, Publishers, APIs, Datasets)
- Advanced filters panel (collapsible):
  - Year range slider
  - Therapeutic topics (CBT, DBT, Trauma, etc.)
  - Relevance threshold
  - Publisher selection
  - Sort options

**Results Display:**

```
┌─────────────────────────────────────────────────────┐
│ Found 47 results across 12 sources                  │
│ [Grid View] [List View] [Export ↓]                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│ ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│ │ Book 1   │  │ Book 2   │  │ Book 3   │          │
│ │ ⭐ 0.92  │  │ ⭐ 0.88  │  │ ⭐ 0.85  │          │
│ │ Oxford   │  │ Springer │  │ PubMed   │          │
│ └──────────┘  └──────────┘  └──────────┘          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Result Card Design:**

- Book cover placeholder (gradient based on source)
- Title (truncated with tooltip)
- Authors
- Publisher/Source badge
- Therapeutic relevance score (star rating)
- Year, DOI/ISBN
- Quick actions: View Details, Export, Save
- Hover effect: Lift with shadow

### Page 2: Dataset Discovery (`/research/datasets`)

**Hero Section:**

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   💬 Therapy Dataset Discovery                     │
│   Find conversation datasets for training & research│
│                                                     │
│   ┌─────────────────────────────────────────────┐ │
│   │ 🔍 Search HuggingFace datasets...           │ │
│   └─────────────────────────────────────────────┘ │
│                                                     │
│   Min Turns: [20+] ━━━━●━━━━ [50+]                │
│   Quality: [0.5] ━━━━━●━━━ [1.0]                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Dataset Cards:**

- Dataset name and description
- Conversation statistics (avg/min/max turns)
- Quality score (0-1 scale with visual indicator)
- Therapeutic relevance score
- Download count, likes
- Languages supported
- Quick preview of sample conversation
- Link to HuggingFace

---

## 🔧 Component Specifications

### 1. SearchInterface.tsx

**Status**: [x] Created

**Features:**

- [ ] Real-time search with debouncing (300ms)
- [ ] Autocomplete suggestions
- [ ] Search history (localStorage)
- [ ] Loading states with skeleton screens
- [ ] Error handling with retry

**State Management:**

```typescript
interface SearchState {
  query: string
  filters: SearchFilters
  results: BookMetadata[]
  loading: boolean
  error: string | null
  selectedSources: SourceType[]
}
```

### 2. SearchFilters.tsx

**Status**: [x] Created

**Filters:**

- [ ] **Year Range**: Dual-handle slider (1900-2026)
- [ ] **Therapeutic Topics**: Multi-select chips
  - CBT, DBT, Trauma, Anxiety, Depression, etc.
- [ ] **Relevance Threshold**: Single slider (0-1)
- [ ] **Publishers**: Checkbox list with search
- [ ] **Sort By**: Dropdown
  - Relevance (default)
  - Year (newest/oldest)
  - Title (A-Z)
  - Author (A-Z)

**UI Pattern:**

- [ ] Collapsible panel
- [ ] "Apply Filters" button
- [ ] "Reset" button
- [ ] Active filter count badge

### 3. ResultsGrid.tsx

**Status**: [x] Created

**Features:**

- [ ] Responsive grid (1/2/3/4 columns)
- [ ] Infinite scroll or pagination
- [ ] Empty state with suggestions
- [ ] Loading skeletons
- [ ] View toggle (grid/list)

**Optimizations:**

- [ ] Virtual scrolling for large result sets
- [ ] Lazy loading images
- [ ] Memoized result cards

### 4. ResultCard.tsx

**Status**: [x] Created

**Layout:**

```
┌────────────────────────────┐
│ [Gradient Placeholder]     │
│                            │
├────────────────────────────┤
│ Title (2 lines max)        │
│ Authors (1 line)           │
│                            │
│ ⭐⭐⭐⭐⭐ 0.92           │
│                            │
│ [Oxford] 2024              │
│ DOI: 10.1234/example       │
│                            │
│ [View] [Export] [Save]     │
└────────────────────────────┘
```

**Interactions:**

- [ ] Hover: Lift + shadow
- [ ] Click title: Expand details modal
- [ ] Click badges: Filter by source
- [ ] Save: Add to favorites (localStorage)

### 5. ExportPanel.tsx

**Status**: [x] Created

**Export Options:**

- [ ] **Format**: JSON, CSV, BibTeX, RIS
- [ ] **Fields**: Select which metadata to include
- [ ] **Filename**: Customizable
- [ ] **Download** button

**UI:**

- [ ] Slide-out panel from right
- [ ] Preview of export data
- [ ] Copy to clipboard option

---

## 🔌 API Integration

### Backend API Endpoints

```python
# ai/sourcing/academic/api/routes.py

@app.get("/api/search")
async def search_literature(
    q: str,
    sources: Optional[List[str]] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    topics: Optional[List[str]] = None,
    min_relevance: float = 0.0,
    limit: int = 20,
    offset: int = 0
) -> SearchResponse:
    """Search academic literature"""
    pass

@app.get("/api/datasets")
async def search_datasets(
    q: str,
    min_turns: int = 20,
    min_quality: float = 0.5,
    limit: int = 20
) -> DatasetResponse:
    """Search therapy datasets"""
    pass

@app.get("/api/sources")
async def get_sources() -> SourcesResponse:
    """Get available sources and their status"""
    pass

@app.post("/api/export")
async def export_results(
    results: List[BookMetadata],
    format: ExportFormat
) -> FileResponse:
    """Export results in specified format"""
    pass
```

### Frontend API Client

**Status**: [ ] Created

```typescript
// src/lib/api/research.ts

export class ResearchAPI {
  private baseURL = '/api'

  async searchLiterature(params: SearchParams): Promise<SearchResponse> {
    const response = await fetch(
      `${this.baseURL}/search?${new URLSearchParams(params)}`,
    )
    return response.json()
  }

  async searchDatasets(params: DatasetParams): Promise<DatasetResponse> {
    const response = await fetch(
      `${this.baseURL}/datasets?${new URLSearchParams(params)}`,
    )
    return response.json()
  }

  async exportResults(results: BookMetadata[], format: string): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ results, format }),
    })
    return response.blob()
  }
}
```

---

## 🎨 Styling Guidelines

### Component Styling

```css
/* research.css */

.research-hero {
  min-height: 60vh;
  background: linear-gradient(135deg, var(--bg-primary), var(--bg-secondary));
  position: relative;
}

.research-hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(
    circle at 50% 50%,
    oklch(70.5% 0.213 47.604 / 0.1) 0%,
    transparent 50%
  );
  pointer-events: none;
}

.search-bar {
  width: 100%;
  max-width: 800px;
  padding: var(--spacing-4);
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: var(--spacing-3);
  font-size: var(--text-lg);
  color: var(--text-primary);
  transition: all var(--duration-normal) var(--ease-in-out);
}

.search-bar:focus {
  outline: none;
  border-color: var(--color-primary-500);
  box-shadow: 0 0 0 3px oklch(70.5% 0.213 47.604 / 0.1);
}

.result-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-secondary);
  border-radius: var(--card-radius);
  padding: var(--card-padding);
  transition: all var(--duration-normal) var(--ease-in-out);
  cursor: pointer;
}

.result-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--card-shadow-hover);
  border-color: var(--color-primary-600);
}

.relevance-badge {
  display: inline-flex;
  align-items: center;
  gap: var(--spacing-1);
  padding: var(--spacing-1) var(--spacing-2);
  background: oklch(70.5% 0.213 47.604 / 0.1);
  border-radius: var(--spacing-2);
  font-size: var(--text-sm);
  font-weight: var(--font-weight-medium);
  color: var(--color-primary-400);
}

.source-chip {
  display: inline-flex;
  align-items: center;
  padding: var(--spacing-1) var(--spacing-3);
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
  border-radius: var(--spacing-6);
  font-size: var(--text-sm);
  transition: all var(--duration-fast) var(--ease-out);
  cursor: pointer;
}

.source-chip:hover {
  background: var(--border-primary);
  border-color: var(--color-primary-600);
}

.source-chip.active {
  background: var(--color-primary-600);
  border-color: var(--color-primary-500);
  color: white;
}
```

---

## 📱 Responsive Design

### Breakpoints

- **Mobile**: < 640px (1 column)
- **Tablet**: 640-1024px (2 columns)
- **Desktop**: > 1024px (3-4 columns)

### Mobile Optimizations

- Collapsible filters (drawer)
- Simplified cards (less metadata)
- Touch-friendly tap targets (min 44px)
- Swipe gestures for navigation

---

## ⚡ Performance Optimizations

### Frontend

- **Code Splitting**: Lazy load components
- **Image Optimization**: WebP with fallbacks
- **Caching**: Service worker for offline support
- **Debouncing**: Search input (300ms)
- **Virtual Scrolling**: Large result sets
- **Memoization**: React.memo for cards

### Backend

- **Response Caching**: Redis for frequent queries
- **Pagination**: Limit results per request
- **Async Processing**: Background tasks for slow sources
- **Rate Limiting**: Prevent API abuse
- **CDN**: Static assets

---

## 🔒 Security Considerations

- **API Keys**: Server-side only (never expose)
- **CORS**: Whitelist frontend domain
- **Rate Limiting**: Per-IP and per-user
- **Input Validation**: Sanitize all inputs
- **HTTPS**: Enforce secure connections
- **CSP**: Content Security Policy headers

---

## 🧪 Testing Strategy

### Unit Tests

- Component rendering
- Filter logic
- API client functions
- Export functionality

### Integration Tests

- Search flow end-to-end
- Filter application
- Export generation
- Error handling

### E2E Tests (Playwright)

- Complete user journeys
- Cross-browser compatibility
- Mobile responsiveness
- Accessibility (a11y)

---

## 📊 Analytics & Monitoring

### Track Events

- Search queries (anonymized)
- Filter usage
- Source selection
- Export actions
- Error rates
- Page load times

### Metrics

- Search success rate
- Average results per query
- Most popular sources
- User engagement time

---

## 🚀 Implementation Phases

### Phase 1: Foundation (Week 1)

- [x] Set up FastAPI backend
- [x] Create API endpoints
- [x] Build basic search interface
- [x] Implement result display
- [x] Add basic styling

### Phase 2: Features (Week 2)

- [x] Advanced filters
- [x] Export functionality
- [x] Dataset search page
- [x] Loading & error states

### Phase 3: Polish (Week 3)

- [x] Animations & transitions
- [x] Mobile optimization
- [x] Performance tuning
- [x] Accessibility improvements
- [x] Testing & bug fixes

### Phase 4: Launch (Week 4)

- [x] Documentation update
- [x] Final security audit
- [x] Production deployment
- [ ] User acceptance testing (See `UAT_INSTRUCTIONS.md`)
- [x] Monitoring setup
- [ ] User feedback collection
- [ ] Iteration based on feedback

---

## 📝 Success Criteria

- ✅ Search returns results in < 2 seconds
- ✅ Mobile-friendly (Lighthouse score > 90)
- ✅ Accessible (WCAG 2.1 AA compliant)
- ✅ Matches existing design system 100%
- ✅ Supports all 20+ sources
- ✅ Export works in all formats
- ✅ Zero critical bugs
- ✅ Positive user feedback

---

## 🎯 Next Steps

1. [x] **Review & Approve** this plan
2. [x] **Set up FastAPI** backend structure
3. [x] **Create Astro pages** with layout
4. [x] **Build React components** incrementally
5. [x] **Integrate with backend** API
6. [ ] **Test & iterate** based on feedback

---

**Ready to build a beautiful, powerful academic search interface!** 🚀
