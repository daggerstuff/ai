# User Acceptance Testing (UAT) Guide

## Academic Sourcing Interface

This guide outlines the steps to verify the functionality and user experience of the new Academic Sourcing Interface.

### 1. Literature Search (`/research`)

**Objective**: Verify search, filtering, and result interaction.

- [ ] **Basic Search**:
  - Enter a term (e.g., "CBT") and press Enter.
  - Verify results appear (mock or real).
  - Verify "Found X results" count is accurate.
- [ ] **Advanced Filters**:
  - Open "Advanced Filters".
  - Set Year Range (e.g., 2020-2024).
  - Select a topic (e.g., "Trauma").
  - Verify results update automatically or upon closure.
- [ ] **Source Selection**:
  - Toggle between "All Sources" and specific publishers (e.g., "Oxford").
  - Verify result badges match selection.
- [ ] **Result Card**:
  - Hover over a card; verify lift animation.
  - Click "View Details" (if URL exists); verify it opens in new tab.
  - Hover over relevance score; check tooltip/visibility.
- [ ] **Export**:
  - Click "Export Results".
  - Select "CSV" or "JSON".
  - Click Download; verify file is generated.

### 2. Dataset Discovery (`/research/datasets`)

**Objective**: Verify dataset searching and metadata display.

- [ ] **Keyword Search**:
  - Search for "depression".
  - Verify generic results or specific dataset matches.
- [ ] **Quality Filters**:
  - Drag "Min Quality" slider to 0.8.
  - Verify low-quality datasets disappear.
- [ ] **Turn Count**:
  - Adjust "Min Turns" to 30.
  - Verify datasets with fewer turns are filtered out.
- [ ] **Card Interaction**:
  - Click a dataset card; verify it links to the source (e.g., HuggingFace).
  - Verify tags (#cbt, #anxiety) are visible.

### 3. Responsiveness & Accessibility

- [ ] **Mobile View**:
  - Resize browser to mobile width (< 640px).
  - Verify Grid becomes 1 column.
  - Verify Filters drawer/modal works correctly.
- [ ] **Keyboard Navigation**:
  - Tab through the interface.
  - Verify focus states are visible (pink rings).
  - ensure Search and Filter controls are reachable.

---

**Report Findings**:
Any issues should be logged in the project issue tracker with the tag `uat-feedback`.
