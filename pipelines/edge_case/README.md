# Pixelated Empathy: Edge Case Generation Pipeline

A standalone, portable pipeline for generating challenging therapy scenarios to train difficult client simulation models.

## 🎯 Features

- **25 Edge Case Categories**: Covering the most challenging therapy scenarios
- **Multiple API Providers**: OpenAI, Anthropic, or local Ollama
- **Jupyter Notebook Interface**: Easy-to-use, step-by-step generation
- **Progress Tracking**: Resumable generation with progress monitoring
- **Quality Assessment**: Automatic evaluation of generated content
- **Training-Ready Output**: Direct integration with training pipelines

## 📦 Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys (if using cloud providers)

For **OpenAI**:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

For **Anthropic**:
```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

For **Ollama** (local):
- Install Ollama: https://ollama.ai
- Pull a model: `ollama pull llama2` or similar
- No API key needed

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook Edge_Case_Generation_Pipeline.ipynb
```

### Option 2: Python Script
```python
from edge_case_generator import EdgeCaseGenerator

# Initialize generator
generator = EdgeCaseGenerator(
    api_provider="openai",  # or "anthropic", "ollama"
    api_key="your_api_key",
    model_name="gpt-3.5-turbo",
    output_dir="output"
)

# Generate prompts (20 per category = 500 total)
prompts = generator.generate_prompts(scenarios_per_category=20)

# Generate conversations
conversations = generator.generate_conversations(prompts, max_conversations=100)

# Create training format
training_data = generator.create_training_format(conversations)

# Generate report
report = generator.generate_summary_report(conversations)
```

## 📊 Edge Case Categories

The pipeline generates scenarios across 25 challenging categories:

### High/Very High Difficulty
- **Suicidality**: Crisis intervention and safety assessment
- **Homicidal Ideation**: Duty to warn and legal considerations
- **Psychotic Episodes**: Reality testing and psychiatric emergencies
- **Child Abuse Reporting**: Mandated reporting situations
- **Severe Dissociation**: Identity switches and memory gaps

### Moderate/High Difficulty
- **Substance Abuse Crisis**: Intoxication and withdrawal
- **Trauma Flashbacks**: Grounding and safety techniques
- **Borderline Crisis**: Emotional dysregulation and self-harm
- **Domestic Violence**: Safety planning and ambivalence
- **Eating Disorders**: Medical risk and body image distortion

### Other Categories
- Paranoid accusations, medication refusal, family conflicts
- Adolescent defiance, couples betrayal, complicated grief
- Cultural conflicts, boundary violations, therapy resistance
- And more...

## 📁 Output Files

The pipeline generates several files:

- `edge_case_prompts.jsonl`: All generated prompts
- `generated_conversations.jsonl`: Raw API responses
- `edge_cases_training_format.jsonl`: **Training-ready data**
- `summary_report.md`: Comprehensive generation report
- `failed_prompts.jsonl`: Failed prompts for retry

## 🔧 Configuration Options

### API Providers
- **OpenAI**: gpt-3.5-turbo, gpt-4, gpt-4-turbo
- **Anthropic**: claude-3-haiku, claude-3-sonnet, claude-3-opus
- **Ollama**: Any local model (llama2, mistral, etc.)

### Generation Settings
- Scenarios per category (1-50)
- Maximum conversations to generate
- Output directory
- Custom model names

## 📈 Performance Tips

### For Speed
- Use OpenAI gpt-3.5-turbo (fastest)
- Local Ollama with smaller models
- Reduce scenarios per category for testing

### For Quality
- Use Claude-3-Sonnet or GPT-4
- Increase scenarios per category
- Review and filter results

### For Cost Efficiency
- Start with small batches (50-100 conversations)
- Use gpt-3.5-turbo instead of gpt-4
- Consider local Ollama for large volumes

## 🔄 Google Colab Usage

To use this in Google Colab:

1. Upload the entire folder to Google Drive
2. Mount Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/edge_case_pipeline_standalone
```

3. Install requirements:
```bash
!pip install -r requirements.txt
```

4. Run the notebook or script

## 🎓 Training Integration

The generated `edge_cases_training_format.jsonl` is ready for training:

```python
# Each line contains:
{
    "prompt": "Therapist statement",
    "response": "Challenging client response", 
    "purpose": "difficult_client",
    "category": "suicidality",
    "difficulty_level": "very_high",
    "expected_challenges": ["crisis_intervention", "safety_assessment"],
    "source": "edge_case_generation"
}
```

## 🐛 Troubleshooting

### Common Issues

**"No conversations generated"**
- Check API key is correct
- Verify model name is valid
- Check rate limits on your API account

**"Connection refused" (Ollama)**
- Ensure Ollama is running: `ollama serve`
- Check model is pulled: `ollama list`
- Verify correct model name

**"Rate limit exceeded"**
- Reduce batch size
- Add delays between requests
- Check your API plan limits

### Resume Failed Generation

If generation fails partway through:
1. Check for `failed_prompts.jsonl`
2. Retry only failed prompts
3. Merge results with successful ones

## 📄 License

This pipeline is part of the Pixelated Empathy project. Use responsibly for educational and research purposes in mental health training.

## 🤝 Contributing

To improve the pipeline:
1. Add new edge case categories
2. Improve conversation extraction
3. Enhance quality assessment
4. Add new API providers

## ⚠️ Ethical Considerations

This tool generates sensitive mental health content for training purposes:

- Use only for legitimate educational/research goals
- Review generated content for appropriateness
- Ensure proper supervision of training data
- Respect privacy and confidentiality principles
- Follow ethical guidelines for AI in healthcare

---

**Ready to generate challenging therapy scenarios? Start with the Jupyter notebook for the best experience!** 