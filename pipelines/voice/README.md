# Pixel Voice API & MCP Server

A **production-ready**, comprehensive API and MCP (Model Context Protocol) server for the Pixel Voice processing pipeline. This system provides enterprise-grade REST API endpoints, MCP tools, web dashboard, and complete deployment infrastructure for voice processing workflows.

**Built with [uv](https://github.com/astral-sh/uv)** - the fast Python package manager and project manager.

## 🚀 Production Features

### 🔒 Enterprise Security & Compliance
- **Multi-tier Authentication**: JWT tokens, API keys, OAuth integration
- **Role-based Access Control**: Admin, Premium, Standard, Read-only roles
- **Rate Limiting & Quotas**: YouTube API compliance, user quotas, abuse prevention
- **Audit Logging**: Comprehensive activity tracking and security monitoring

### 📊 Production Infrastructure
- **Kubernetes Deployment**: Full K8s manifests with auto-scaling and health checks
- **Monitoring & Alerting**: Prometheus metrics, Grafana dashboards, alert rules
- **Database Persistence**: PostgreSQL with migrations and backup procedures
- **Caching Layer**: Redis for sessions, rate limiting, and performance optimization
- **CI/CD Pipeline**: GitHub Actions with automated testing and deployment

### 🌐 User-Friendly Interface
- **Web Dashboard**: Complete pipeline management and monitoring interface
- **Real-time Updates**: WebSocket connections for live job status and notifications
- **Usage Analytics**: Detailed usage tracking and quota management
- **Interactive Documentation**: Comprehensive API docs with live examples

### ⚡ Performance & Scalability
- **Horizontal Scaling**: Auto-scaling based on CPU/memory usage
- **Load Balancing**: Multiple API replicas with intelligent request distribution
- **Async Processing**: Non-blocking pipeline execution with job queuing
- **Resource Optimization**: Efficient memory and CPU usage patterns

### FastAPI Server
- **Pipeline Management**: Execute individual stages or complete pipeline jobs
- **Real-time Monitoring**: WebSocket connections for live job status updates
- **Data Access**: Retrieve processed data (transcripts, features, dialogue pairs, etc.)
- **Job Control**: Create, monitor, and cancel pipeline jobs
- **Health Monitoring**: System status and health check endpoints
- **User Management**: Authentication, authorization, and quota management

### MCP Server
- **Pipeline Tools**: Execute pipeline stages and jobs through MCP tools
- **Status Monitoring**: Check job status and system health
- **Data Retrieval**: Access processed data through MCP interface
- **YouTube Transcription**: Direct YouTube URL transcription tool

## Quick Start

### 1. Prerequisites

Install [uv](https://github.com/astral-sh/uv) if you haven't already:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### 2. Setup

Run the automated setup script:

```bash
# This will install uv (if needed), create virtual environment, and install dependencies
python setup.py
```

Or manually:

```bash
# Create virtual environment
uv venv .venv

# Install the project and dependencies
uv pip install -e .

# Install pipeline dependencies
uv pip install -r requirements/pixel_voice_pipeline.txt
```

### 3. Configuration

The setup script creates a default `.env` file, or create one manually:

```env
PIXEL_VOICE_ENV=development
PIXEL_VOICE_DEBUG=true
PIXEL_VOICE_API_HOST=0.0.0.0
PIXEL_VOICE_API_PORT=8000
PIXEL_VOICE_MCP_HOST=localhost
PIXEL_VOICE_MCP_PORT=8001
```

### 4. Start the Services

```bash
# Start the API Server
uv run python start_api.py

# Start the MCP Server (in another terminal)
uv run python start_mcp.py
```

Or if you have activated the virtual environment:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start services
python start_api.py
python start_mcp.py
```

## 🏭 Production Deployment

### Quick Production Setup

```bash
# 1. Build and deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# 2. Or deploy to Kubernetes
kubectl apply -f k8s/

# 3. Access the services
# API: https://your-domain.com
# Dashboard: https://your-domain.com/dashboard
# Docs: https://your-domain.com/docs
# Metrics: https://your-domain.com/metrics
```

### Production Features

- **🔐 Security**: JWT authentication, API keys, rate limiting, audit logs
- **📊 Monitoring**: Prometheus metrics, Grafana dashboards, health checks
- **🚀 Scaling**: Kubernetes auto-scaling, load balancing, high availability
- **💾 Persistence**: PostgreSQL database, Redis caching, data backups
- **🌐 Web UI**: Complete dashboard for pipeline management and monitoring
- **🔄 CI/CD**: Automated testing, building, and deployment pipelines

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete production deployment guide.

## API Endpoints

### Core Endpoints

- `GET /` - Health check and system information
- `GET /pipeline/status` - Overall pipeline status
- `POST /transcribe` - Transcribe YouTube URL (simple interface)

### Job Management

- `POST /pipeline/jobs` - Create and start a new pipeline job
- `GET /pipeline/jobs` - List all jobs (with optional status filter)
- `GET /pipeline/jobs/{job_id}` - Get specific job information
- `DELETE /pipeline/jobs/{job_id}` - Cancel a running job

### Stage Execution

- `POST /pipeline/stages/execute` - Execute a single pipeline stage

### Data Access

- `GET /data/{data_type}/latest` - Get latest processed data
- `GET /data/{data_type}/files` - List all data files of a type

Supported data types: `transcripts`, `features`, `dialogue_pairs`, `therapeutic_pairs`, `consistency`, `optimized`

### Real-time Updates

- `WebSocket /ws` - Real-time job status updates and system events

## MCP Tools

The MCP server provides the following tools:

### Pipeline Execution
- `run_pipeline_stage` - Execute a single pipeline stage
- `run_full_pipeline` - Execute a complete pipeline job
- `transcribe_youtube` - Transcribe audio from YouTube URL

### Monitoring & Control
- `get_job_status` - Check status of a specific job
- `list_jobs` - List all pipeline jobs
- `cancel_job` - Cancel a running job
- `get_pipeline_status` - Get overall system status

### Data Access
- `get_latest_data` - Retrieve latest processed data
- `list_data_files` - List available data files

## Usage Examples

### API Examples

#### Create a Pipeline Job
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post("http://localhost:8000/pipeline/jobs", json={
        "job_name": "Process YouTube Video",
        "stages": ["batch_transcription", "feature_extraction", "dialogue_construction"],
        "input_data": {"youtube_url": "https://youtube.com/watch?v=example"},
        "config_overrides": {"whisper_model": "large-v2"}
    })
    job_id = response.text.strip('"')
    print(f"Job created: {job_id}")
```

#### Monitor Job Status
```python
response = await client.get(f"http://localhost:8000/pipeline/jobs/{job_id}")
job_info = response.json()
print(f"Status: {job_info['status']}, Progress: {job_info['progress']:.1%}")
```

#### Get Latest Transcripts
```python
response = await client.get("http://localhost:8000/data/transcripts/latest")
transcripts = response.json()
print(f"Found {len(transcripts)} transcript segments")
```

#### Run Examples
```bash
# Run API client example
uv run python examples/api_client_example.py

# Run MCP client example
uv run python examples/mcp_client_example.py

# Or use the example runner
uv run python run_examples.py
```

### MCP Examples

When using the MCP server with an MCP client:

#### Transcribe YouTube Video
```json
{
  "tool": "transcribe_youtube",
  "arguments": {
    "youtube_url": "https://youtube.com/watch?v=example",
    "language": "en",
    "whisper_model": "large-v2"
  }
}
```

#### Run Full Pipeline
```json
{
  "tool": "run_full_pipeline",
  "arguments": {
    "job_name": "Complete Voice Processing",
    "stages": ["audio_quality_control", "batch_transcription", "feature_extraction"],
    "input_data": {"source": "youtube_playlist"}
  }
}
```

## Configuration

The system uses a hierarchical configuration system with the following sources (in order of precedence):

1. Environment variables (prefixed with `PIXEL_VOICE_`)
2. `.env` file
3. Default configuration values

### Key Configuration Options

- `PIXEL_VOICE_ENV` - Environment (development/production)
- `PIXEL_VOICE_DEBUG` - Enable debug mode
- `PIXEL_VOICE_API_HOST` - API server host
- `PIXEL_VOICE_API_PORT` - API server port
- `PIXEL_VOICE_MCP_HOST` - MCP server host
- `PIXEL_VOICE_MCP_PORT` - MCP server port

## Pipeline Stages

The system supports the following pipeline stages:

1. **audio_quality_control** - Audio quality assessment and segmentation
2. **batch_transcription** - WhisperX transcription with diarization
3. **transcription_filtering** - Quality filtering of transcripts
4. **feature_extraction** - Emotion, sentiment, and text feature extraction
5. **personality_clustering** - Personality and emotion clustering
6. **dialogue_construction** - Dialogue pair construction
7. **dialogue_validation** - ML-based validation
8. **therapeutic_generation** - Therapeutic pair generation
9. **voice_consistency** - Voice quality consistency assessment
10. **voice_filtering** - Advanced data filtering and optimization

## Development

### Development Setup
```bash
# Install development dependencies
uv pip install -e .[dev]

# Install pre-commit hooks (optional)
uv run pre-commit install
```

### Running Tests
```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=pixel_voice

# Run specific test file
uv run pytest tests/test_api.py
```

### Code Quality
```bash
# Format code
uv run black pixel_voice/

# Check code style
uv run flake8 pixel_voice/

# Type checking
uv run mypy pixel_voice/

# Run all quality checks
uv run pre-commit run --all-files
```

### Adding New Endpoints
1. Add request/response models to `api/models.py`
2. Implement endpoint in `api/server.py`
3. Add corresponding MCP tool in `mcp_server.py` if needed
4. Update documentation
5. Add tests for new functionality

### Virtual Environment Management
```bash
# Create new virtual environment
uv venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install project in development mode
uv pip install -e .[dev]

# Deactivate virtual environment
deactivate
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the parent directory is in Python path
2. **Permission Errors**: Check file permissions for data directories
3. **Port Conflicts**: Modify port configuration if defaults are in use
4. **Pipeline Failures**: Check individual stage logs in the `logs/` directory

### Logging

Logs are written to:
- API Server: Console output and uvicorn logs
- MCP Server: Console output
- Pipeline Stages: Individual log files in `logs/` directory

### Health Checks

- API Health: `GET /`
- Pipeline Status: `GET /pipeline/status`
- MCP Status: Use the `get_pipeline_status` tool

## License

This project is part of the Pixel Voice pipeline system.
