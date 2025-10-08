# Getting Started with Pixelated Empathy AI

**Welcome to Pixelated Empathy AI!** This guide will help you get started with our enterprise-grade conversational AI dataset and processing system.

## Table of Contents

1. [Overview](#overview)
2. [Account Setup](#account-setup)
3. [First Steps](#first-steps)
4. [Basic Operations](#basic-operations)
5. [Understanding Quality Tiers](#understanding-quality-tiers)
6. [Common Use Cases](#common-use-cases)
7. [Next Steps](#next-steps)

---

## Overview

Pixelated Empathy AI provides access to **2.59 million high-quality therapeutic conversations** with advanced quality validation, real-time processing, and comprehensive analytics. Our system is designed for:

- **Researchers** studying conversational AI and mental health
- **Developers** building empathy-aware applications
- **Organizations** training therapeutic AI systems
- **Data Scientists** analyzing conversation patterns and quality

### What You Get Access To

- **2.59M+ Conversations**: Professionally curated therapeutic dialogues
- **5 Quality Tiers**: From basic to research-grade conversations
- **Real-Time Processing**: Submit jobs and get live status updates
- **Advanced Search**: Find conversations by content, quality, and metadata
- **Multiple Export Formats**: JSONL, CSV, Parquet, HuggingFace, OpenAI
- **Quality Validation**: Real NLP-based assessment (not fake scores)

---

## Account Setup

### Step 1: Register for an Account

1. Visit [https://api.pixelatedempathy.com/register](https://api.pixelatedempathy.com/register)
2. Fill out the registration form with:
   - Your name and email address
   - Organization details
   - Intended use case (research, commercial, educational)
   - Brief description of your project

### Step 2: Email Verification

1. Check your email for a verification message
2. Click the verification link to confirm your account
3. You'll be redirected to a confirmation page

### Step 3: Application Review

1. Our team will review your application within 24-48 hours
2. You'll receive an email notification when approved
3. Research and educational use cases are typically approved quickly
4. Commercial applications may require additional information

### Step 4: Get Your API Key

1. Once approved, log into your account dashboard
2. Navigate to the "API Keys" section
3. Click "Generate New Key" to create your first API key
4. **Important**: Copy and store your API key securely - it won't be shown again

---

## First Steps

### Test Your API Access

Once you have your API key, test your access:

#### Using cURL
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.pixelatedempathy.com/v1/datasets
```

#### Using Python
```python
import requests

headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}

response = requests.get(
    'https://api.pixelatedempathy.com/v1/datasets', 
    headers=headers
)

print(response.json())
```

### Expected Response
```json
{
    "success": true,
    "data": {
        "datasets": [
            {
                "name": "priority_complete_fixed",
                "description": "Priority conversations with complete processing",
                "conversations": 297917,
                "quality_score": 0.624,
                "tiers": ["basic", "standard", "professional"]
            }
        ],
        "total": 3
    },
    "message": "Datasets retrieved successfully"
}
```

---

## Basic Operations

### 1. Explore Available Datasets

Start by understanding what datasets are available:

```python
from pixelated_empathy_api import PixelatedEmpathyAPI

api = PixelatedEmpathyAPI("YOUR_API_KEY")

# List all datasets
datasets = api.list_datasets()
for dataset in datasets:
    print(f"Dataset: {dataset['name']}")
    print(f"  Conversations: {dataset['conversations']:,}")
    print(f"  Quality Score: {dataset['quality_score']:.3f}")
    print(f"  Tiers: {', '.join(dataset['tiers'])}")
    print()
```

### 2. Browse Conversations

Get a sample of conversations to understand the data structure:

```python
# Get 5 professional-tier conversations
conversations = api.get_conversations(
    tier="professional", 
    limit=5
)

print(f"Found {len(conversations['conversations'])} conversations")

# Look at the first conversation
first_conv = conversations['conversations'][0]
print(f"Conversation ID: {first_conv['id']}")
print(f"Quality Score: {first_conv['quality_score']}")
print(f"Message Count: {first_conv['message_count']}")
```

### 3. Examine a Full Conversation

Get the complete details of a specific conversation:

```python
# Get full conversation details
conversation = api.get_conversation("conv_000001")

print("Messages:")
for i, message in enumerate(conversation['messages']):
    print(f"{i+1}. {message['role']}: {message['content']}")

print("\nQuality Metrics:")
metrics = conversation['quality_metrics']
for metric, score in metrics.items():
    print(f"  {metric}: {score:.3f}")
```

### 4. Search for Specific Content

Find conversations about specific topics:

```python
# Search for anxiety-related conversations
results = api.search_conversations(
    "anxiety therapy techniques",
    filters={
        "tier": "professional",
        "min_quality": 0.7
    },
    limit=10
)

print(f"Found {results['total_matches']} matching conversations")
for result in results['results']:
    print(f"- {result['conversation_id']}: {result['snippet']}")
```

---

## Understanding Quality Tiers

Our conversations are organized into 5 quality tiers based on real NLP analysis:

### **Research Tier** (Quality: 0.82+)
- **Count**: 11,730 conversations
- **Use Case**: Academic research, publication-quality studies
- **Features**: Highest therapeutic accuracy, clinical compliance
- **Access**: Requires research verification

### **Clinical Tier** (Quality: 0.80+)
- **Count**: 33,739 conversations
- **Use Case**: Clinical training, therapeutic applications
- **Features**: DSM-5 compliant, professional therapeutic techniques
- **Access**: Requires clinical credentials or institutional approval

### **Professional Tier** (Quality: 0.74+)
- **Count**: 89,870 conversations
- **Use Case**: Commercial applications, professional training
- **Features**: High therapeutic accuracy, emotional authenticity
- **Access**: Standard commercial license

### **Standard Tier** (Quality: 0.69+)
- **Count**: 179,740 conversations
- **Use Case**: General development, prototyping
- **Features**: Good conversation coherence, basic therapeutic patterns
- **Access**: Standard API access

### **Basic Tier** (Quality: 0.62+)
- **Count**: 2,277,144 conversations
- **Use Case**: Training data, bulk processing, experimentation
- **Features**: Validated conversations, basic quality assurance
- **Access**: All users

### Choosing the Right Tier

```python
# Get quality metrics to understand tier distribution
metrics = api.get_quality_metrics()

print("Quality Distribution:")
for tier, data in metrics['tier_metrics'].items():
    print(f"{tier.title()}: {data['count']:,} conversations "
          f"(avg quality: {data['average_quality']:.3f})")
```

---

## Common Use Cases

### Use Case 1: Training a Chatbot

**Goal**: Train a therapeutic chatbot using high-quality conversations

**Steps**:
1. **Select appropriate tier**: Professional or Clinical for therapeutic applications
2. **Filter by quality**: Set minimum quality threshold (e.g., 0.75)
3. **Export training data**: Use OpenAI format for fine-tuning
4. **Validate quality**: Use our quality validation API

```python
# Export professional conversations for training
export_info = api.export_data(
    dataset="professional_datasets_final",
    format="openai",
    tier="professional",
    min_quality=0.75
)

print(f"Export initiated: {export_info['export_id']}")
print(f"Estimated size: {export_info['estimated_size']}")
print(f"Download URL: {export_info['download_url']}")
```

### Use Case 2: Research Analysis

**Goal**: Analyze conversation patterns for academic research

**Steps**:
1. **Access research tier**: Highest quality conversations
2. **Search by topic**: Find conversations about specific conditions
3. **Analyze quality metrics**: Study therapeutic effectiveness
4. **Export for analysis**: Use Parquet format for data science tools

```python
# Search for depression-related research conversations
research_data = api.search_conversations(
    "depression cognitive behavioral therapy",
    filters={
        "tier": "research",
        "min_quality": 0.85
    }
)

# Export for analysis
export_info = api.export_data(
    dataset="research_conversations",
    format="parquet",
    tier="research"
)
```

### Use Case 3: Quality Assessment

**Goal**: Validate the quality of your own conversations

**Steps**:
1. **Prepare your conversation**: Format as required
2. **Submit for validation**: Use quality validation API
3. **Review results**: Analyze quality metrics and recommendations
4. **Improve based on feedback**: Apply recommendations

```python
# Validate your conversation
my_conversation = {
    "id": "my_conv_001",
    "messages": [
        {"role": "user", "content": "I'm feeling overwhelmed at work."},
        {"role": "assistant", "content": "That sounds really challenging. Can you tell me more about what's making you feel overwhelmed?"}
    ],
    "quality_score": 0.0,
    "tier": "unknown"
}

validation = api.validate_conversation_quality(my_conversation)
print(f"Overall Quality: {validation['validation_results']['overall_quality']:.3f}")
print(f"Tier Classification: {validation['tier_classification']}")
print("Recommendations:")
for rec in validation['recommendations']:
    print(f"- {rec}")
```

### Use Case 4: Bulk Processing

**Goal**: Process large amounts of conversation data

**Steps**:
1. **Submit processing job**: Specify dataset and processing type
2. **Monitor progress**: Check job status regularly
3. **Retrieve results**: Download processed data when complete

```python
# Submit a large processing job
job = api.submit_processing_job(
    dataset_name="priority_complete_fixed",
    processing_type="quality_validation",
    parameters={
        "tier_filter": "standard",
        "min_quality": 0.7,
        "output_format": "jsonl"
    }
)

print(f"Job submitted: {job['job_id']}")

# Wait for completion (this will poll automatically)
final_status = api.wait_for_job(job['job_id'])
print(f"Job completed: {final_status['status']}")
print(f"Results: {final_status['results']}")
```

---

## Next Steps

### Explore Advanced Features

1. **Read the API Documentation**: [Complete API Reference](../api/complete_api_documentation.md)
2. **Try the Interactive Docs**: Visit https://api.pixelatedempathy.com/docs
3. **Join the Community**: Connect with other users on our Discord
4. **Follow Best Practices**: Review our [Best Practices Guide](best_practices.md)

### Get Support

- **Documentation**: Browse our comprehensive guides
- **API Reference**: Detailed endpoint documentation
- **Community Forum**: Ask questions and share experiences
- **Email Support**: api-support@pixelatedempathy.com (response within 24 hours)
- **Status Page**: Check system status at status.pixelatedempathy.com

### Stay Updated

- **Changelog**: Monitor API updates and new features
- **Newsletter**: Subscribe for monthly updates and tips
- **GitHub**: Follow our repositories for code examples
- **Blog**: Read case studies and technical deep-dives

---

## Quick Reference

### Essential API Endpoints
- **List Datasets**: `GET /v1/datasets`
- **Get Conversations**: `GET /v1/conversations`
- **Search**: `POST /v1/search`
- **Quality Validation**: `POST /v1/quality/validate`
- **Export Data**: `POST /v1/export`

### Quality Tiers (by minimum score)
- **Research**: 0.82+ (11,730 conversations)
- **Clinical**: 0.80+ (33,739 conversations)  
- **Professional**: 0.74+ (89,870 conversations)
- **Standard**: 0.69+ (179,740 conversations)
- **Basic**: 0.62+ (2,277,144 conversations)

### Rate Limits
- **Free Tier**: 100 requests/hour
- **Research**: 1,000 requests/hour
- **Commercial**: 10,000 requests/hour
- **Enterprise**: Custom limits

### Export Formats
- **JSONL**: Standard conversation format
- **CSV**: Human-readable spreadsheet format
- **Parquet**: Efficient data analysis format
- **HuggingFace**: ML framework compatibility
- **OpenAI**: Fine-tuning format

---

**Ready to get started?** [Set up your account](https://api.pixelatedempathy.com/register) and begin exploring our 2.59 million therapeutic conversations today!

*Need help? Contact us at api-support@pixelatedempathy.com*
