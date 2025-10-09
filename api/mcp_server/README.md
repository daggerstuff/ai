# MCP (Management Control Panel) Server

## Overview

The MCP Server is a comprehensive management control panel for agent interaction endpoints, designed to orchestrate the 6-stage TechDeck-Python pipeline integration process. It provides agent registration, task delegation, pipeline orchestration, and real-time communication capabilities.

## Architecture

### Core Components

```
MCP Server
├── Agent Management
│   ├── Registration & Discovery
│   ├── Health Monitoring
│   └── Load Balancing
├── Task Orchestration
│   ├── Task Queue
│   ├── Status Tracking
│   └── Retry Mechanisms
├── Pipeline Engine
│   ├── 6-Stage Process Orchestration
│   ├── Stage Validation
│   └── Progress Tracking
├── Communication Layer
│   ├── RESTful APIs
│   ├── WebSocket Support
│   └── Event Broadcasting
├── Security & Auth
│   ├── Authentication
│   ├── Authorization
│   └── Rate Limiting
└── Monitoring & Logging
    ├── Performance Metrics
    ├── Error Tracking
    └── Audit Logs
```

## API Endpoints

### Agent Management

#### Register Agent
```http
POST /api/v1/agents/register
Content-Type: application/json

{
  "name": "agent-name",
  "type": "worker|orchestrator|validator",
  "capabilities": ["task-execution", "data-processing"],
  "endpoints": {
    "task": "http://agent:8080/task",
    "status": "http://agent:8080/status",
    "health": "http://agent:8080/health"
  },
  "resources": {
    "cpu": 4,
    "memory": "8GB",
    "gpu": true
  }
}
```

#### Discover Agents
```http
GET /api/v1/agents
Accept: application/json

Response:
{
  "agents": [
    {
      "id": "agent-123",
      "name": "data-processor",
      "type": "worker",
      "status": "active",
      "capabilities": ["data-processing"],
      "last_heartbeat": "2023-07-27T10:00:00Z",
      "current_load": 0.75
    }
  ]
}
```

### Task Management

#### Submit Task
```http
POST /api/v1/tasks
Content-Type: application/json

{
  "pipeline_id": "pipeline-123",
  "stage": 1,
  "type": "data-ingestion|transformation|analysis|validation|processing|output",
  "payload": {
    "data_source": "s3://bucket/data.csv",
    "parameters": {
      "format": "csv",
      "delimiter": ","
    }
  },
  "priority": "high|medium|low",
  "timeout": 300,
  "retry_count": 3
}
```

#### Get Task Status
```http
GET /api/v1/tasks/{task_id}
Accept: application/json

Response:
{
  "task_id": "task-123",
  "pipeline_id": "pipeline-123",
  "stage": 1,
  "status": "pending|running|completed|failed|retrying",
  "progress": 0.75,
  "assigned_agent": "agent-456",
  "started_at": "2023-07-27T10:00:00Z",
  "updated_at": "2023-07-27T10:05:00Z",
  "logs": [
    {
      "timestamp": "2023-07-27T10:00:00Z",
      "level": "info",
      "message": "Task started"
    }
  ],
  "error": null
}
```

### Pipeline Orchestration

#### Create Pipeline
```http
POST /api/v1/pipelines
Content-Type: application/json

{
  "name": "data-processing-pipeline",
  "description": "6-stage data processing pipeline",
  "stages": [
    {
      "name": "data-ingestion",
      "type": "data-ingestion",
      "required": true,
      "timeout": 300
    },
    {
      "name": "data-transformation",
      "type": "transformation",
      "required": true,
      "timeout": 600
    },
    {
      "name": "data-analysis",
      "type": "analysis",
      "required": true,
      "timeout": 900
    },
    {
      "name": "data-validation",
      "type": "validation",
      "required": true,
      "timeout": 300
    },
    {
      "name": "data-processing",
      "type": "processing",
      "required": true,
      "timeout": 1200
    },
    {
      "name": "output-generation",
      "type": "output",
      "required": true,
      "timeout": 300
    }
  ],
  "dependencies": [
    {"from": 0, "to": 1},
    {"from": 1, "to": 2},
    {"from": 2, "to": 3},
    {"from": 3, "to": 4},
    {"from": 4, "to": 5}
  ]
}
```

#### Execute Pipeline
```http
POST /api/v1/pipelines/{pipeline_id}/execute
Content-Type: application/json

{
  "input_data": {
    "source": "s3://bucket/input-data",
    "format": "json"
  },
  "parameters": {
    "batch_size": 1000,
    "parallel_workers": 4
  },
  "notification": {
    "webhook": "https://example.com/webhook",
    "email": "admin@example.com"
  }
}
```

### WebSocket Events

#### Connection
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://mcp-server:8080/ws');

// Authentication message
ws.send(JSON.stringify({
  type: 'auth',
  token: 'jwt-token-here'
}));
```

#### Event Types
```javascript
// Task Status Update
{
  "type": "task_update",
  "task_id": "task-123",
  "status": "running",
  "progress": 0.5,
  "timestamp": "2023-07-27T10:00:00Z"
}

// Agent Status Update
{
  "type": "agent_update",
  "agent_id": "agent-456",
  "status": "active",
  "current_load": 0.75,
  "timestamp": "2023-07-27T10:00:00Z"
}

// Pipeline Progress
{
  "type": "pipeline_progress",
  "pipeline_id": "pipeline-123",
  "current_stage": 2,
  "completed_stages": [0, 1],
  "progress": 0.33,
  "timestamp": "2023-07-27T10:00:00Z"
}

// Error Notification
{
  "type": "error",
  "task_id": "task-123",
  "error": "Task execution failed",
  "details": {
    "code": "EXECUTION_ERROR",
    "message": "Invalid input data format"
  },
  "timestamp": "2023-07-27T10:00:00Z"
}
```

## Integration with Flask Service

### Authentication
The MCP Server integrates with the existing Flask authentication system:
- Uses JWT tokens for API authentication
- Supports role-based access control (RBAC)
- Integrates with existing user management

### Data Flow
```
Flask Service → MCP Server → Agents
     ↓              ↓            ↓
  User Auth → Task Creation → Task Execution
  Data Store → Pipeline Mgmt → Result Processing
```

### Shared Resources
- **Database**: MongoDB for agent registry and task tracking
- **Cache**: Redis for session management and caching
- **Message Queue**: Redis/RabbitMQ for task distribution
- **File Storage**: Shared S3/Cloud Storage for data exchange

## Security Features

### Authentication
- JWT-based authentication
- API key support for agent registration
- OAuth2 integration for external services

### Authorization
- Role-based access control (RBAC)
- Resource-level permissions
- Agent-specific access tokens

### Rate Limiting
- API endpoint rate limiting
- Task submission quotas
- Concurrent connection limits

### Data Protection
- TLS encryption for all communications
- Secure token storage
- Audit logging for all operations

## Monitoring & Logging

### Metrics
- Task execution times
- Agent availability and performance
- Pipeline success rates
- Error rates and types

### Logging
- Structured logging with JSON format
- Log levels: DEBUG, INFO, WARN, ERROR
- Log retention and rotation policies
- Centralized log aggregation

### Health Checks
- Agent health monitoring
- Service availability checks
- Database connectivity verification
- Resource usage monitoring

## Deployment

### Docker Configuration
```yaml
# docker-compose.yml
version: '3.8'
services:
  mcp-server:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MONGODB_URI=mongodb://mongo:27017/mcp
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=your-secret-key
      - FLASK_API_URL=http://flask-service:5000
    depends_on:
      - mongo
      - redis
    volumes:
      - ./logs:/app/logs

  mongo:
    image: mongo:4.4
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  mongo-data:
  redis-data:
```

### Environment Variables
```bash
# Database Configuration
MONGODB_URI=mongodb://localhost:27017/mcp
REDIS_URL=redis://localhost:6379

# Security
JWT_SECRET=your-jwt-secret-key
API_KEY=your-api-key

# Service Configuration
FLASK_API_URL=http://flask-service:5000
SERVER_PORT=8080
WS_PORT=8080

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
METRICS_ENABLED=true

# Performance
MAX_AGENTS=100
MAX_CONCURRENT_TASKS=50
TASK_TIMEOUT=3600
```

## Development

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m mcp_server.main

# Run tests
pytest tests/

# Run with hot reload
python -m uvicorn mcp_server.main:app --reload --host 0.0.0.0 --port 8080
```

### Testing
```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=mcp_server --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.