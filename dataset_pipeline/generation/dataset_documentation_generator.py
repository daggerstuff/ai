"""
Dataset Documentation Generator

Automated generation of comprehensive dataset documentation including
sources, licenses, usage constraints, and metadata summaries.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetSource:
    """Dataset source information."""

    name: str
    url: str
    description: str
    license: str
    license_url: str | None = None
    citation: str | None = None
    version: str | None = None
    download_date: str | None = None
    contact: str | None = None


@dataclass
class DatasetMetadata:
    """Comprehensive dataset metadata."""

    name: str
    description: str
    version: str
    created_date: str
    file_count: int
    total_size: int
    format: str
    sources: list[DatasetSource] = field(default_factory=list)
    usage_constraints: list[str] = field(default_factory=list)
    ethical_considerations: list[str] = field(default_factory=list)
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class DatasetDocumentationGenerator:
    """Generate comprehensive dataset documentation."""

    def __init__(self):
        self.logger = get_logger(__name__)

        # Common dataset sources and their information
        self.known_sources = {
            "huggingface": {
                "base_url": "https://huggingface.co/datasets/",
                "license_info": "Varies by dataset - check individual dataset pages",
                "citation_format": "@misc{{{name}, author = {{Hugging Face}}, title = {{{title}}}, year = {{2023}}, url = {{https://huggingface.co/datasets/{name}}}}}",
            },
            "openai": {
                "base_url": "https://openai.com/",
                "license_info": "Custom OpenAI License",
                "citation_format": "@misc{{{name}, author = {{OpenAI}}, title = {{{title}}}, year = {{2023}}, url = {{https://openai.com/}}}",
            },
            "anthropic": {
                "base_url": "https://www.anthropic.com/",
                "license_info": "Custom Anthropic License",
                "citation_format": "@misc{{{name}, author = {{Anthropic}}, title = {{{title}}}, year = {{2023}}, url = {{https://www.anthropic.com/}}}",
            },
        }

        # License templates
        self.license_templates = {
            "MIT": {
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT",
                "commercial_use": True,
                "modification": True,
                "distribution": True,
                "private_use": True,
            },
            "Apache-2.0": {
                "name": "Apache License 2.0",
                "url": "https://www.apache.org/licenses/LICENSE-2.0",
                "commercial_use": True,
                "modification": True,
                "distribution": True,
                "private_use": True,
            },
            "CC-BY-4.0": {
                "name": "Creative Commons Attribution 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "commercial_use": True,
                "modification": True,
                "distribution": True,
                "private_use": True,
            },
            "CC-BY-NC-4.0": {
                "name": "Creative Commons Attribution-NonCommercial 4.0",
                "url": "https://creativecommons.org/licenses/by-nc/4.0/",
                "commercial_use": False,
                "modification": True,
                "distribution": True,
                "private_use": True,
            },
        }

        logger.info("DatasetDocumentationGenerator initialized")

    def generate_documentation(
        self,
        dataset_path: str,
        metadata: DatasetMetadata,
        output_dir: str | None = None,
    ) -> dict[str, str]:
        """Generate comprehensive dataset documentation."""
        if not output_dir:
            output_dir = os.path.join(dataset_path, "documentation")

        os.makedirs(output_dir, exist_ok=True)

        generated_files = {}

        # Generate README.md
        readme_path = os.path.join(output_dir, "README.md")
        self._generate_readme(metadata, readme_path)
        generated_files["readme"] = readme_path

        # Generate LICENSE file
        license_path = os.path.join(output_dir, "LICENSE.md")
        self._generate_license_file(metadata, license_path)
        generated_files["license"] = license_path

        # Generate CITATION.bib
        citation_path = os.path.join(output_dir, "CITATION.bib")
        self._generate_citation_file(metadata, citation_path)
        generated_files["citation"] = citation_path

        # Generate metadata.json
        metadata_path = os.path.join(output_dir, "metadata.json")
        self._generate_metadata_json(metadata, metadata_path)
        generated_files["metadata"] = metadata_path

        # Generate USAGE.md
        usage_path = os.path.join(output_dir, "USAGE.md")
        self._generate_usage_guide(metadata, usage_path)
        generated_files["usage"] = usage_path

        # Generate ETHICS.md
        ethics_path = os.path.join(output_dir, "ETHICS.md")
        self._generate_ethics_guide(metadata, ethics_path)
        generated_files["ethics"] = ethics_path

        # Generate dataset card (YAML)
        card_path = os.path.join(output_dir, "dataset_card.yaml")
        self._generate_dataset_card(metadata, card_path)
        generated_files["dataset_card"] = card_path

        logger.info(f"Generated documentation in: {output_dir}")
        return generated_files

    def _generate_readme(self, metadata: DatasetMetadata, output_path: str):
        """Generate README.md file."""
        readme_content = f"""# {metadata.name}

## Description

{metadata.description}

## Dataset Information

- **Version**: {metadata.version}
- **Created**: {metadata.created_date}
- **Format**: {metadata.format}
- **File Count**: {metadata.file_count:,}
- **Total Size**: {self._format_size(metadata.total_size)}

## Sources

"""

        for source in metadata.sources:
            readme_content += f"""### {source.name}

- **URL**: {source.url}
- **Description**: {source.description}
- **License**: {source.license}
"""
            if source.license_url:
                readme_content += f"- **License URL**: {source.license_url}\n"
            if source.citation:
                readme_content += f"- **Citation**: {source.citation}\n"
            if source.version:
                readme_content += f"- **Version**: {source.version}\n"
            if source.download_date:
                readme_content += f"- **Downloaded**: {source.download_date}\n"
            readme_content += "\n"

        if metadata.tags:
            readme_content += f"""## Tags

{', '.join(f'`{tag}`' for tag in metadata.tags)}

"""

        if metadata.quality_metrics:
            readme_content += """## Quality Metrics

"""
            for metric, value in metadata.quality_metrics.items():
                readme_content += f"- **{metric.replace('_', ' ').title()}**: {value}\n"
            readme_content += "\n"

        readme_content += """## Usage

See [USAGE.md](USAGE.md) for detailed usage instructions and examples.

## License

See [LICENSE.md](LICENSE.md) for licensing information.

## Ethics and Considerations

See [ETHICS.md](ETHICS.md) for ethical considerations and usage guidelines.

## Citation

If you use this dataset, please cite it using the information in [CITATION.bib](CITATION.bib).

## Contact

For questions or issues regarding this dataset, please refer to the individual source contacts listed above.
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        logger.info(f"Generated README.md: {output_path}")

    def _generate_license_file(self, metadata: DatasetMetadata, output_path: str):
        """Generate LICENSE.md file."""
        license_content = f"""# License Information for {metadata.name}

This dataset is a compilation of multiple sources, each with their own licensing terms.

## Individual Source Licenses

"""

        for source in metadata.sources:
            license_content += f"""### {source.name}

- **License**: {source.license}
"""
            if source.license_url:
                license_content += f"- **License URL**: {source.license_url}\n"

            # Add license details if known
            if source.license in self.license_templates:
                template = self.license_templates[source.license]
                license_content += f"- **Full Name**: {template['name']}\n"
                license_content += f"- **Official URL**: {template['url']}\n"
                license_content += f"- **Commercial Use**: {'✅ Allowed' if template['commercial_use'] else '❌ Not Allowed'}\n"
                license_content += f"- **Modification**: {'✅ Allowed' if template['modification'] else '❌ Not Allowed'}\n"
                license_content += f"- **Distribution**: {'✅ Allowed' if template['distribution'] else '❌ Not Allowed'}\n"

            license_content += "\n"

        license_content += """## Usage Requirements

When using this dataset, you must comply with ALL individual source licenses. This means:

1. **Attribution**: Provide proper attribution for each source as specified in their licenses
2. **Commercial Use**: Only use commercially if ALL sources allow commercial use
3. **Modification**: Only modify if ALL sources allow modification
4. **Distribution**: Only redistribute if ALL sources allow distribution

## Disclaimer

The creators of this compiled dataset are not responsible for license violations.
Users are solely responsible for ensuring compliance with all applicable licenses.

## Recommended Citation

See CITATION.bib for recommended citation format that includes all sources.
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(license_content)

        logger.info(f"Generated LICENSE.md: {output_path}")

    def _generate_citation_file(self, metadata: DatasetMetadata, output_path: str):
        """Generate CITATION.bib file."""
        citation_content = f"""% Citation for {metadata.name}
% Generated on {datetime.now().strftime('%Y-%m-%d')}

% Main dataset citation
@dataset{{{metadata.name.lower().replace(' ', '_')},
  title = {{{metadata.name}}},
  description = {{{metadata.description}}},
  version = {{{metadata.version}}},
  year = {{{datetime.now().year}}},
  url = {{Local compilation}},
  note = {{Compiled dataset from multiple sources}}
}}

% Individual source citations
"""

        for i, source in enumerate(metadata.sources):
            source_key = f"{metadata.name.lower().replace(' ', '_')}_source_{i+1}"
            citation_content += f"""
@misc{{{source_key},
  title = {{{source.name}}},
  description = {{{source.description}}},
  url = {{{source.url}}},
  license = {{{source.license}}}"""

            if source.version:
                citation_content += f",\n  version = {{{source.version}}}"
            if source.download_date:
                citation_content += (
                    f",\n  note = {{Downloaded on {source.download_date}}}"
                )

            citation_content += "\n}\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(citation_content)

        logger.info(f"Generated CITATION.bib: {output_path}")

    def _generate_metadata_json(self, metadata: DatasetMetadata, output_path: str):
        """Generate metadata.json file."""
        metadata_dict = {
            "name": metadata.name,
            "description": metadata.description,
            "version": metadata.version,
            "created_date": metadata.created_date,
            "file_count": metadata.file_count,
            "total_size": metadata.total_size,
            "format": metadata.format,
            "tags": metadata.tags,
            "quality_metrics": metadata.quality_metrics,
            "usage_constraints": metadata.usage_constraints,
            "ethical_considerations": metadata.ethical_considerations,
            "sources": [
                {
                    "name": source.name,
                    "url": source.url,
                    "description": source.description,
                    "license": source.license,
                    "license_url": source.license_url,
                    "citation": source.citation,
                    "version": source.version,
                    "download_date": source.download_date,
                    "contact": source.contact,
                }
                for source in metadata.sources
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated metadata.json: {output_path}")

    def _generate_usage_guide(self, metadata: DatasetMetadata, output_path: str):
        """Generate USAGE.md file."""
        usage_content = f"""# Usage Guide for {metadata.name}

## Loading the Dataset

### Python Example

```python
import json
import pandas as pd
from pathlib import Path

# Load JSON dataset
def load_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Load JSONL dataset
def load_jsonl_dataset(dataset_path):
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Example usage
dataset = load_dataset('path/to/dataset.json')
print(f"Loaded {{len(dataset)}} items")
```

## Data Format

The dataset is in **{metadata.format}** format with the following structure:

"""

        if metadata.format.lower() == "json":
            usage_content += """```json
{
  "conversations": [
    {
      "id": "unique_conversation_id",
      "messages": [
        {
          "role": "user",
          "content": "User message content"
        },
        {
          "role": "assistant",
          "content": "Assistant response content"
        }
      ],
      "metadata": {
        "source": "source_name",
        "quality_score": 0.85,
        "tags": ["tag1", "tag2"]
      }
    }
  ]
}
```
"""

        usage_content += """## Usage Constraints

"""

        for constraint in metadata.usage_constraints:
            usage_content += f"- {constraint}\n"

        usage_content += """
## Best Practices

1. **Data Validation**: Always validate the data structure before processing
2. **Quality Filtering**: Consider filtering by quality scores if available
3. **Source Attribution**: Maintain source attribution when using subsets
4. **License Compliance**: Ensure compliance with all source licenses

## Example Processing Pipeline

```python
def process_conversations(dataset):
    processed = []

    for conversation in dataset.get('conversations', []):
        # Validate conversation structure
        if 'messages' not in conversation:
            continue

        # Filter by quality if available
        quality = conversation.get('metadata', {}).get('quality_score', 0)
        if quality < 0.7:  # Adjust threshold as needed
            continue

        # Process messages
        processed_messages = []
        for message in conversation['messages']:
            if message.get('content'):
                processed_messages.append({
                    'role': message['role'],
                    'content': message['content'].strip()
                })

        if processed_messages:
            processed.append({
                'id': conversation.get('id'),
                'messages': processed_messages,
                'source': conversation.get('metadata', {}).get('source')
            })

    return processed

# Usage
processed_data = process_conversations(dataset)
```

## Integration Examples

### With Hugging Face Datasets

```python
from datasets import Dataset

def convert_to_hf_dataset(conversations):
    # Convert to Hugging Face format
    examples = []
    for conv in conversations:
        for i in range(0, len(conv['messages']), 2):
            if i + 1 < len(conv['messages']):
                examples.append({
                    'input': conv['messages'][i]['content'],
                    'output': conv['messages'][i + 1]['content'],
                    'source': conv.get('source', 'unknown')
                })

    return Dataset.from_list(examples)

hf_dataset = convert_to_hf_dataset(processed_data)
```

### With PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]
        # Implement tokenization logic here
        return {
            'input_ids': torch.tensor([1, 2, 3]),  # Placeholder
            'attention_mask': torch.tensor([1, 1, 1])  # Placeholder
        }

# Usage
dataset = ConversationDataset(processed_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Troubleshooting

### Common Issues

1. **Encoding Errors**: Ensure files are read with UTF-8 encoding
2. **Memory Issues**: For large datasets, consider streaming or chunking
3. **Format Inconsistencies**: Validate data structure before processing

### Performance Tips

1. Use streaming for large datasets
2. Implement caching for repeated access
3. Consider parallel processing for data transformation
4. Use appropriate data types to minimize memory usage

## Support

For technical issues or questions about this dataset, please refer to the individual source documentation and contact information listed in the README.md file.
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(usage_content)

        logger.info(f"Generated USAGE.md: {output_path}")

    def _generate_ethics_guide(self, metadata: DatasetMetadata, output_path: str):
        """Generate ETHICS.md file."""
        ethics_content = f"""# Ethical Considerations for {metadata.name}

## Overview

This document outlines important ethical considerations when using this dataset.
Users are responsible for ensuring ethical and responsible use.

## Ethical Considerations

"""

        for consideration in metadata.ethical_considerations:
            ethics_content += f"- {consideration}\n"

        ethics_content += """
## Bias and Fairness

### Potential Biases

This dataset may contain various forms of bias including but not limited to:

- **Demographic Bias**: Underrepresentation of certain demographic groups
- **Cultural Bias**: Overrepresentation of certain cultural perspectives
- **Temporal Bias**: Data may reflect attitudes from specific time periods
- **Source Bias**: Bias inherent in the original data sources

### Mitigation Strategies

1. **Bias Assessment**: Regularly assess model outputs for biased behavior
2. **Diverse Evaluation**: Test on diverse demographic groups and use cases
3. **Bias Correction**: Implement bias correction techniques during training
4. **Ongoing Monitoring**: Continuously monitor deployed systems for bias

## Privacy and Consent

### Data Privacy

- All data sources should have appropriate privacy protections
- Personal identifiable information (PII) should be removed or anonymized
- Users should implement additional privacy safeguards as needed

### Consent Considerations

- Verify that data collection had appropriate consent mechanisms
- Respect any usage limitations specified by original data providers
- Consider additional consent requirements for your specific use case

## Responsible Use Guidelines

### Recommended Uses

- Research and development of conversational AI systems
- Training models for therapeutic and mental health applications
- Academic research on dialogue systems and human-computer interaction
- Development of empathetic AI assistants

### Discouraged Uses

- Creating systems that could cause psychological harm
- Developing manipulative or deceptive AI systems
- Training models without appropriate safety measures
- Commercial use without proper license compliance

### Prohibited Uses

- Creating systems designed to harm individuals or groups
- Developing AI for surveillance without consent
- Training models for illegal activities
- Using data in ways that violate source licenses

## Safety Considerations

### Model Safety

1. **Output Filtering**: Implement filters to prevent harmful outputs
2. **Safety Testing**: Thoroughly test for potential safety issues
3. **Human Oversight**: Maintain human oversight in deployment
4. **Feedback Mechanisms**: Implement user feedback and reporting systems

### Deployment Safety

1. **Gradual Rollout**: Deploy gradually with monitoring
2. **User Education**: Educate users about system limitations
3. **Clear Disclaimers**: Provide clear disclaimers about AI nature
4. **Emergency Procedures**: Have procedures for handling safety issues

## Transparency and Accountability

### Documentation Requirements

- Maintain clear documentation of data sources and processing
- Document any modifications or filtering applied to the data
- Keep records of model training and evaluation procedures
- Provide clear information to end users about system capabilities and limitations

### Accountability Measures

- Establish clear responsibility chains for AI system behavior
- Implement logging and auditing capabilities
- Provide mechanisms for user feedback and complaints
- Regular review and update of ethical guidelines

## Compliance and Legal Considerations

### Regulatory Compliance

- Ensure compliance with applicable data protection regulations (GDPR, CCPA, etc.)
- Follow industry-specific regulations for healthcare, finance, etc.
- Comply with AI governance frameworks in your jurisdiction
- Maintain compliance with export control regulations if applicable

### Legal Responsibilities

- Respect intellectual property rights of all data sources
- Comply with terms of service and license agreements
- Understand liability implications of AI system deployment
- Maintain appropriate insurance and legal protections

## Reporting and Feedback

### Issue Reporting

If you identify ethical concerns, bias, or safety issues with this dataset or models trained on it:

1. Document the issue thoroughly with examples
2. Report to the dataset maintainers (see contact information in README.md)
3. Consider reporting to relevant regulatory bodies if appropriate
4. Share findings with the research community when possible

### Continuous Improvement

- Participate in community discussions about ethical AI development
- Share best practices and lessons learned
- Contribute to the development of better ethical guidelines
- Support research into AI safety and ethics

## Resources

### Additional Reading

- [Partnership on AI Tenets](https://www.partnershiponai.org/tenets/)
- [IEEE Standards for Ethical AI](https://standards.ieee.org/industry-connections/ec/autonomous-systems.html)
- [AI Ethics Guidelines Global Inventory](https://inventory.algorithmwatch.org/)
- [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/)

### Professional Organizations

- Association for Computing Machinery (ACM) Code of Ethics
- IEEE Computer Society Code of Ethics
- Partnership on AI
- AI Now Institute

## Conclusion

Ethical AI development is an ongoing responsibility that extends beyond dataset creation to model training, deployment, and maintenance. Users of this dataset are encouraged to prioritize ethical considerations throughout their AI development lifecycle.

This document should be regularly reviewed and updated as ethical standards and best practices evolve.

---

*Last updated: {datetime.now().strftime('%Y-%m-%d')}*
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ethics_content)

        logger.info(f"Generated ETHICS.md: {output_path}")

    def _generate_dataset_card(self, metadata: DatasetMetadata, output_path: str):
        """Generate dataset_card.yaml file."""
        card_data = {
            "dataset_info": {
                "dataset_name": metadata.name,
                "description": metadata.description,
                "version": metadata.version,
                "created_date": metadata.created_date,
                "format": metadata.format,
                "size_info": {
                    "file_count": metadata.file_count,
                    "total_size_bytes": metadata.total_size,
                    "total_size_human": self._format_size(metadata.total_size),
                },
            },
            "sources": [
                {
                    "name": source.name,
                    "url": source.url,
                    "license": source.license,
                    "description": source.description,
                }
                for source in metadata.sources
            ],
            "tags": metadata.tags,
            "quality_metrics": metadata.quality_metrics,
            "usage_constraints": metadata.usage_constraints,
            "ethical_considerations": metadata.ethical_considerations,
            "recommended_use_cases": [
                "Conversational AI training",
                "Dialogue system research",
                "Empathetic AI development",
                "Mental health AI applications",
            ],
            "limitations": [
                "May contain biases from source data",
                "Quality varies across sources",
                "Requires careful ethical consideration for deployment",
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(card_data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Generated dataset_card.yaml: {output_path}")

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def create_metadata_from_dataset(
        self, dataset_path: str, name: str, description: str
    ) -> DatasetMetadata:
        """Create metadata by analyzing dataset directory."""
        file_count = 0
        total_size = 0
        formats = set()

        for root, _dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_count += 1
                total_size += os.path.getsize(file_path)
                formats.add(Path(file).suffix.lower())

        primary_format = (
            max(
                formats, key=lambda x: sum(1 for f in Path(dataset_path).rglob(f"*{x}"))
            )
            if formats
            else "unknown"
        )

        return DatasetMetadata(
            name=name,
            description=description,
            version="1.0.0",
            created_date=datetime.now().strftime("%Y-%m-%d"),
            file_count=file_count,
            total_size=total_size,
            format=(
                primary_format.lstrip(".").upper()
                if primary_format != "unknown"
                else "Mixed"
            ),
        )


# Example usage
if __name__ == "__main__":
    generator = DatasetDocumentationGenerator()

    # Example metadata
    metadata = DatasetMetadata(
        name="Pixelated Empathy Dataset",
        description="A comprehensive dataset for training empathetic conversational AI",
        version="1.0.0",
        created_date="2025-08-26",
        file_count=1000,
        total_size=500000000,  # 500MB
        format="JSON",
        tags=["conversational-ai", "empathy", "mental-health", "dialogue"],
        usage_constraints=[
            "Must comply with all source licenses",
            "Intended for research and therapeutic applications",
            "Commercial use requires additional review",
        ],
        ethical_considerations=[
            "Contains sensitive mental health content",
            "May reflect biases from source data",
            "Requires careful handling of privacy concerns",
        ],
    )

    # Add example sources
    metadata.sources.append(
        DatasetSource(
            name="Empathetic Dialogues",
            url="https://huggingface.co/datasets/empathetic_dialogues",
            description="Dataset of empathetic conversations",
            license="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
        )
    )

    # Generate documentation
    files = generator.generate_documentation("./test_dataset", metadata)
