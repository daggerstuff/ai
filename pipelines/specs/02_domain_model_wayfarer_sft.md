# Domain Model: Wayfarer-2-12B SFT Pipeline

## Core Entities & Data Structures

### 1. Model
- **Attributes:**
  - `model_path: str`
  - `config: dict`
  - `tokenizer: Tokenizer`
- **Description:** Loaded Wayfarer-2-12B model object, compatible with Unsloth and Lightning.ai Studio.

### 2. RawSample
- **Attributes:**
  - `source: str` (dataset path or identifier)
  - `data: dict` (raw sample content)
- **Validation:**
  - Must contain required fields (e.g., prompt, response).
  - No null or empty values.

### 3. CleanSample
- **Attributes:**
  - `prompt: str`
  - `response: str`
  - `metadata: dict` (optional, e.g., tags, source)
- **Validation:**
  - Prompt and response must be non-empty, trimmed, and free of control characters.
  - No duplicates (exact or near-duplicate).

### 4. ChatMLSample
- **Attributes:**
  - `chatml: str` (serialized ChatML format)
  - `metadata: dict`
- **Validation:**
  - Must conform to ChatML schema (roles, message structure).
  - All required fields present.

### 5. TokenizedDataset
- **Attributes:**
  - `input_ids: List[List[int]]`
  - `attention_mask: List[List[int]]`
  - `metadata: dict`
- **Validation:**
  - All sequences within model’s max length.
  - Padding and special tokens as required by tokenizer.

### 6. ComplianceReport
- **Attributes:**
  - `issues: List[ComplianceIssue]`
  - `summary: dict`
- **ComplianceIssue:**
  - `sample_id: str`
  - `issue_type: str` (privacy, bias, etc.)
  - `description: str`
  - `severity: str` (info, warning, critical)
- **Validation:**
  - All flagged issues must reference valid sample IDs.
  - Summary includes counts by type/severity.

## Relationships

- **RawSample** → cleaned to **CleanSample**
- **CleanSample** → converted to **ChatMLSample**
- **ChatMLSample** → tokenized to **TokenizedDataset**
- **ChatMLSample** → checked for compliance, results in **ComplianceReport**

## State Transitions

1. **Ingested** (RawSample)
2. **Cleaned** (CleanSample)
3. **ChatML-Converted** (ChatMLSample)
4. **Tokenized** (TokenizedDataset)
5. **Compliance-Checked** (ComplianceReport)

## Validation Rules

- All user inputs (paths, configs) must be validated for existence and format.
- Data at each stage must pass entity-specific validation before proceeding.
- Errors must be logged with sufficient context for debugging/audit.

## Glossary

- **ChatML:** Standardized message format for chat-based LLMs.
- **Compliance Issue:** Any privacy, bias, or security concern detected in data.
- **Tokenizer:** Object responsible for converting text to model input IDs.

---