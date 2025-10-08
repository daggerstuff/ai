# Pixelated Empathy AI - API Documentation

**Version:** 1.0.0  
**Generated:** 2025-08-03T21:11:11.151244  
**Base URL:** https://api.pixelatedempathy.com/v1

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Dataset Access](#dataset_access)
- [Quality Metrics](#quality_metrics)
- [Processing](#processing)
- [Export Formats](#export_formats)
- [Search](#search)
- [Statistics](#statistics)
- [Error Handling](#error_handling)
- [Rate Limits](#rate_limits)
- [Sdk Examples](#sdk_examples)

---

## Overview {#overview}

### Description

RESTful API for accessing the Pixelated Empathy AI dataset and processing system

### Version

1.0.0

### Base Url

https://api.pixelatedempathy.com/v1

### Protocols

- HTTPS

### Data Formats

- JSON
- JSONL
- CSV
- Parquet

### Authentication

API Key based authentication

### Rate Limits

1000 requests per hour for standard users

### Key Features

- Dataset access and filtering
- Quality metrics and validation
- Real-time processing capabilities
- Multiple export formats
- Advanced search and filtering
- Comprehensive statistics and analytics

### Supported Operations

- GET - Retrieve data and information
- POST - Submit processing requests
- PUT - Update configurations
- DELETE - Remove data (with appropriate permissions)

## Authentication {#authentication}

### Method

API Key Authentication

### Header

Authorization: Bearer YOUR_API_KEY

### Obtaining Key

#### Registration

Register at https://api.pixelatedempathy.com/register

#### Verification

Email verification required

#### Approval

Manual approval for research and commercial use

#### Key Generation

API key generated upon approval

### Key Management

#### Rotation

Keys should be rotated every 90 days

#### Storage

Store keys securely, never in code repositories

#### Sharing

Never share API keys with unauthorized users

#### Revocation

Keys can be revoked immediately if compromised

### Authentication Example

#### Curl

curl -H 'Authorization: Bearer YOUR_API_KEY' https://api.pixelatedempathy.com/v1/datasets

#### Python

```
import requests

headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}

response = requests.get('https://api.pixelatedempathy.com/v1/datasets', headers=headers)
```

#### Javascript

```
const headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
};

fetch('https://api.pixelatedempathy.com/v1/datasets', { headers })
    .then(response => response.json())
    .then(data => console.log(data));
```

## Dataset Access {#dataset_access}

### Endpoints

#### List Datasets

##### Method

GET

##### Path

/datasets

##### Description

List all available datasets

##### Parameters

###### Tier

Filter by tier (1, 2, 3)

###### Quality Min

Minimum quality score (0.0-1.0)

###### Limit

Number of results to return (default: 100)

###### Offset

Offset for pagination (default: 0)

##### Response

###### Datasets

- **id**: priority_1
- **name**: Priority Dataset Tier 1
- **description**: High-priority therapeutic conversations
- **conversation_count**: 102594
- **average_quality**: 0.637
- **tier**: 1
- **created_at**: 2024-08-01T00:00:00Z

###### Total

25

###### Limit

100

###### Offset

0

#### Get Dataset

##### Method

GET

##### Path

/datasets/{dataset_id}

##### Description

Get detailed information about a specific dataset

##### Parameters

###### Dataset Id

Unique dataset identifier

##### Response

###### Id

priority_1

###### Name

Priority Dataset Tier 1

###### Description

High-priority therapeutic conversations

###### Conversation Count

102594

###### Average Quality

0.637

###### Tier

1

###### Metadata

####### Source

Multiple therapeutic conversation sources

####### Processing Date

2024-08-01T00:00:00Z

####### Quality Validation

Real NLP-based validation

#### Get Conversations

##### Method

GET

##### Path

/datasets/{dataset_id}/conversations

##### Description

Get conversations from a specific dataset

##### Parameters

###### Dataset Id

Unique dataset identifier

###### Quality Min

Minimum quality score

###### Quality Max

Maximum quality score

###### Limit

Number of conversations to return

###### Offset

Offset for pagination

###### Format

Response format (json, jsonl, csv)

##### Response

###### Conversations

- **id**: conv_12345
- **turns**: [{'speaker': 'user', 'text': "I've been feeling really anxious lately..."}, {'speaker': 'assistant', 'text': 'I understand that anxiety can be overwhelming...'}]
- **quality_score**: 0.75
- **metadata**: {'tier': 1, 'topic': 'anxiety', 'length': 8}

###### Total

102594

###### Limit

100

###### Offset

0

## Quality Metrics {#quality_metrics}

### Endpoints

#### Get Quality Metrics

##### Method

GET

##### Path

/quality/metrics

##### Description

Get overall quality metrics for all datasets

##### Response

###### Overall Metrics

####### Average Quality

0.687

####### Total Conversations

2592223

####### High Quality Count

1234567

####### Quality Distribution

######## 0.8-1.0

456789

######## 0.6-0.8

777778

######## 0.4-0.6

357656

######## 0.0-0.4

0

#### Validate Conversation

##### Method

POST

##### Path

/quality/validate

##### Description

Validate quality of a conversation

##### Request Body

###### Conversation

####### Turns

- **speaker**: user
- **text**: I'm feeling depressed
- **speaker**: assistant
- **text**: I'm sorry to hear you're feeling this way...

##### Response

###### Quality Score

0.75

###### Quality Breakdown

####### Therapeutic Accuracy

0.8

####### Conversation Coherence

0.85

####### Emotional Authenticity

0.7

####### Clinical Compliance

0.75

####### Personality Consistency

0.65

####### Language Quality

0.9

###### Recommendations

- Consider more specific therapeutic techniques
- Enhance emotional validation

## Processing {#processing}

### Endpoints

#### Submit Processing Job

##### Method

POST

##### Path

/processing/jobs

##### Description

Submit a new processing job

##### Request Body

###### Job Type

dataset_processing

###### Input Data

####### Source

url_or_file_reference

####### Format

jsonl

###### Processing Options

####### Quality Validation

True

####### Deduplication

True

####### Tier Assignment

True

##### Response

###### Job Id

job_12345

###### Status

queued

###### Estimated Completion

2024-08-03T15:30:00Z

###### Created At

2024-08-03T14:00:00Z

#### Get Job Status

##### Method

GET

##### Path

/processing/jobs/{job_id}

##### Description

Get status of a processing job

##### Response

###### Job Id

job_12345

###### Status

processing

###### Progress

0.65

###### Processed Items

6500

###### Total Items

10000

###### Estimated Completion

2024-08-03T15:30:00Z

###### Results

####### Output Dataset Id

processed_12345

####### Quality Report

Available upon completion

## Export Formats {#export_formats}

### Supported Formats

#### Jsonl

##### Description

JSON Lines format for streaming processing

##### Mime Type

application/jsonl

##### Use Cases

- Model training
- Data processing
- Streaming

#### Parquet

##### Description

Columnar storage format for analytics

##### Mime Type

application/parquet

##### Use Cases

- Data analysis
- Big data processing
- Analytics

#### Csv

##### Description

Comma-separated values for human readability

##### Mime Type

text/csv

##### Use Cases

- Manual review
- Spreadsheet analysis
- Simple processing

#### Huggingface

##### Description

HuggingFace datasets format

##### Mime Type

application/json

##### Use Cases

- HuggingFace model training
- Transformers library

### Endpoints

#### Export Dataset

##### Method

POST

##### Path

/export

##### Description

Export dataset in specified format

##### Request Body

###### Dataset Ids

- priority_1
- priority_2

###### Format

jsonl

###### Filters

####### Quality Min

0.7

####### Tier

- 1
- 2

###### Split

####### Train

0.7

####### Validation

0.15

####### Test

0.15

##### Response

###### Export Id

export_12345

###### Status

processing

###### Estimated Completion

2024-08-03T16:00:00Z

#### Download Export

##### Method

GET

##### Path

/export/{export_id}/download

##### Description

Download completed export

##### Response

Binary file download or redirect to download URL

## Search {#search}

### Endpoints

#### Search Conversations

##### Method

GET

##### Path

/search/conversations

##### Description

Search conversations using full-text search

##### Parameters

###### Q

Search query string

###### Filters

JSON object with filters

###### Limit

Number of results (default: 20)

###### Offset

Offset for pagination

##### Response

###### Results

- **conversation_id**: conv_12345
- **relevance_score**: 0.95
- **snippet**: ...feeling anxious lately...
- **metadata**: {'quality_score': 0.75, 'tier': 1, 'dataset': 'priority_1'}

###### Total

1234

###### Query Time

0.045

## Statistics {#statistics}

### Endpoints

#### Get Statistics

##### Method

GET

##### Path

/statistics

##### Description

Get comprehensive dataset statistics

##### Response

###### Total Conversations

2592223

###### Average Quality

0.687

###### Tier Distribution

####### Tier 1

297917

####### Tier 2

1234567

####### Tier 3

1059739

###### Quality Distribution

####### High Quality

1234567

####### Medium Quality

777778

####### Low Quality

579878

## Error Handling {#error_handling}

### Error Format

#### Error

##### Code

ERROR_CODE

##### Message

Human readable error message

##### Details

Additional error details

##### Timestamp

2024-08-03T14:00:00Z

### Status Codes

#### 200

Success

#### 400

Bad Request - Invalid parameters

#### 401

Unauthorized - Invalid API key

#### 403

Forbidden - Insufficient permissions

#### 404

Not Found - Resource not found

#### 429

Too Many Requests - Rate limit exceeded

#### 500

Internal Server Error

## Rate Limits {#rate_limits}

### Limits

#### Standard

1000 requests per hour

#### Premium

10000 requests per hour

#### Enterprise

Unlimited with fair use

### Headers

#### X-Ratelimit-Limit

Request limit per hour

#### X-Ratelimit-Remaining

Remaining requests

#### X-Ratelimit-Reset

Reset time (Unix timestamp)

## Sdk Examples {#sdk_examples}

### Python

#### Installation

pip install pixelated-empathy-sdk

#### Example

```
from pixelated_empathy import PixelatedEmpathyClient

client = PixelatedEmpathyClient(api_key="YOUR_API_KEY")

# Get datasets
datasets = client.get_datasets()

# Get conversations
conversations = client.get_conversations(
    dataset_id="priority_1",
    quality_min=0.7,
    limit=100
)

# Validate conversation quality
quality = client.validate_conversation(conversation_data)
```

### Javascript

#### Installation

npm install pixelated-empathy-sdk

#### Example

```
const PixelatedEmpathy = require('pixelated-empathy-sdk');

const client = new PixelatedEmpathy({
    apiKey: 'YOUR_API_KEY'
});

// Get datasets
const datasets = await client.getDatasets();

// Get conversations
const conversations = await client.getConversations({
    datasetId: 'priority_1',
    qualityMin: 0.7,
    limit: 100
});
```

