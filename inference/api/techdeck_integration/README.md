 # TechDeck-Python Pipeline Integration Flask API Service

A comprehensive Flask API service that bridges TechDeck React frontend with
Python dataset pipeline processing, featuring HIPAA++ compliance, real-time bias
detection, WebSocket progress tracking, and enterprise-grade error handling.

## 🎯 Overview

This Flask service provides a robust REST API layer for integrating TechDeck's
React-based frontend with Python dataset processing pipelines. It handles
dataset management, pipeline orchestration, bias detection, real-time progress
tracking, and comprehensive analytics while maintaining strict HIPAA++
compliance and sub-50ms response times for critical operations.

## 🏗️ Architecture

### Core Components

- **Flask Application Factory**: Modular application structure with blueprints
- **JWT Authentication**: Role-based access control with rate limiting
- **Six-Stage Pipeline**: Data ingestion → validation → standardization → bias
  detection → processing → output
- **WebSocket Integration**: Real-time progress tracking and event broadcasting
- **Redis Integration**: Caching, pub/sub, and session management
- **HIPAA++ Compliance**: Encrypted data handling and audit logging

### Technology Stack

- **Backend**: Flask 3.0+ with Python 3.11+
- **Authentication**: PyJWT with role-based access control
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Caching**: Redis with asyncio support
- **File Storage**: Secure multi-format support (CSV, JSON, JSONL, Parquet)
- **Bias Detection**: Integrated bias detection service
- **Monitoring**: Structured logging with audit trails

## 📁 Project Structure

```
techdeck_integration/
├── app.py                    # Main Flask application factory
├── config.py                 # Environment-based configuration
├── main.py                   # Application entry point
├── requirements.txt          # Python dependencies
├── routes/                   # API route blueprints
│   ├── datasets.py          # Dataset CRUD operations
│   ├── pipeline.py          # Pipeline orchestration
│   ├── standardization.py   # Data standardization
│   ├── validation.py        # Data validation with bias detection
│   ├── analytics.py         # Usage and performance analytics
│   └── system.py            # System health and configuration
├── services/                 # Business logic layer
│   ├── dataset_service.py   # Dataset operations service
│   └── pipeline_service.py  # Pipeline orchestration service
├── integration/              # External service integrations
│   ├── redis_client.py      # Redis connection and operations
│   ├── pipeline_orchestrator.py # Six-stage pipeline coordination
│   └── bias_detection.py    # Bias detection service integration
├── auth/                     # Authentication and authorization
│   └── middleware.py        # JWT validation and rate limiting
├── error_handling/           # Comprehensive error management
│   └── custom_errors.py     # Custom exception types
├── utils/                    # Utility modules
│   ├── logger.py            # HIPAA-compliant structured logging
│   ├── validation.py        # Input validation and sanitization
│   ├── encryption.py        # Data encryption utilities
│   └── file_handler.py      # Secure file operations
├── websocket/                # Real-time communication
│   └── progress_tracker.py  # WebSocket progress tracking
└── tests/                    # Test suites
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis server
- PostgreSQL database
- Bias detection service (optional)

### Installation

1. **Clone and setup environment:**

```bash
cd ai/api/techdeck_integration
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Initialize database:**

```bash
python -c "from app import create_app; app = create_app(); app.app_context().push()"
```

5. **Start the service:**

```bash
python main.py
```

The service will start on `http://localhost:5000` by default.

## 📡 API Endpoints

### Dataset Management

- `GET /api/datasets` - List datasets with pagination
- `POST /api/datasets` - Upload new dataset
- `GET /api/datasets/{id}` - Get dataset details
- `PUT /api/datasets/{id}` - Update dataset metadata
- `DELETE /api/datasets/{id}` - Delete dataset
- `GET /api/datasets/{id}/download` - Download dataset

### Pipeline Operations

- `GET /api/pipeline/config` - Get pipeline configuration
- `POST /api/pipeline/config` - Update pipeline configuration
- `POST /api/pipeline/execute` - Execute pipeline
- `GET /api/pipeline/status/{operation_id}` - Get operation status
- `POST /api/pipeline/cancel/{operation_id}` - Cancel operation

### Data Processing

- `POST /api/standardization/execute` - Standardize data format
- `POST /api/validation/execute` - Validate data quality
- `GET /api/validation/schema/{format}` - Get validation schema

### Analytics & Monitoring

- `GET /api/analytics/usage` - Usage metrics
- `GET /api/analytics/performance` - Performance metrics
- `GET /api/analytics/pipeline` - Pipeline analytics
- `GET /api/system/health` - System health check
- `GET /api/system/metrics` - System metrics

### WebSocket (Real-time)

- `ws://localhost:5000/ws/progress/{operation_id}` - Progress updates

## 🔐 Authentication

The service uses JWT-based authentication with role-based access control:

### Roles

- **admin**: Full access to all endpoints
- **user**: Standard dataset and pipeline operations
- **viewer**: Read-only access

### Authentication Flow

1. Obtain JWT token from authentication service
2. Include token in `Authorization: Bearer <token>` header
3. Token is validated with configurable expiration and refresh

### Rate Limiting

- Default: 100 requests per minute per user
- Configurable per endpoint and user role
- Redis-backed for distributed deployments

## 📊 Bias Detection Integration

The service integrates with bias detection services to ensure ethical AI
processing:

### Supported Bias Types

- Gender bias
- Racial/ethnic bias
- Age bias
- Socioeconomic bias
- Geographic bias
- Disability bias
- Religious bias
- Sexual orientation bias

### Bias Detection Features

- Real-time bias scoring (0.0-1.0 scale)
- Configurable thresholds (default: 0.7)
- Detailed bias metrics and recommendations
- Compliance validation
- Audit trail logging

## 🔄 Six-Stage Pipeline

The pipeline processes data through six coordinated stages:

1. **Data Ingestion**: Secure file upload and validation
2. **Data Validation**: Quality checks and bias detection
3. **Standardization**: Format normalization and schema validation
4. **Bias Detection**: Comprehensive bias analysis
5. **Processing**: Core dataset transformation
6. **Output Generation**: Result compilation and delivery

Each stage provides real-time progress updates via WebSocket.

## 🛡️ Security Features

### HIPAA++ Compliance

- End-to-end encryption for sensitive data
- Audit logging for all data access
- PII detection and sanitization
- Secure file storage with integrity verification
- Compliance reporting and monitoring

### Input Validation

- Comprehensive input sanitization
- File type and size validation
- SQL injection prevention
- XSS protection
- Rate limiting and DDoS protection

### Data Protection

- Encryption at rest and in transit
- Secure key management
- Data minimization principles
- Right to deletion implementation
- Cross-origin resource sharing (CORS) protection

## 📈 Performance Optimization

### Response Time Targets

- Critical operations: <50ms
- Standard API calls: <200ms
- File uploads: <5s for 100MB files
- Pipeline execution: Progress updates every 2 seconds

### Optimization Features

- Redis caching for frequently accessed data
- Database query optimization
- Connection pooling
- Async processing for long-running operations
- CDN integration for static assets

## 🔍 Monitoring & Observability

### Logging

- Structured JSON logging
- HIPAA-compliant audit trails
- Performance metrics collection
- Error tracking with context
- Security event logging

### Metrics

- API response times
- Pipeline execution duration
- Error rates and types
- Resource utilization
- User activity patterns

### Health Checks

- Database connectivity
- Redis availability
- External service dependencies
- Disk space and memory usage
- Certificate validity

## 🧪 Testing

### Test Coverage

- Unit tests: >80% coverage
- Integration tests: All API endpoints
- Performance tests: Load and stress testing
- Security tests: Vulnerability scanning
- Bias detection tests: Fairness validation

### Test Commands

```bash
# Run all tests
python -m pytest tests/ -v --cov=.

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/performance/ -v
```

## 🚢 Deployment

### Docker Support

```bash
# Build container
docker build -t techdeck-flask-api .

# Run with docker-compose
docker-compose up -d
```

### Environment Configuration

- Development, staging, and production configs
- Environment variable management
- Secret rotation support
- Feature flags for gradual rollouts

### Scaling

- Horizontal scaling with load balancers
- Database read replicas
- Redis clustering
- CDN integration
- Auto-scaling policies

## 📚 Documentation

### API Documentation

- OpenAPI 3.0 specification
- Interactive Swagger UI
- Postman collection
- Code examples for all endpoints

### Architecture Documentation

- System design diagrams
- Data flow documentation
- Security architecture
- Deployment guides

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch
3. Implement with tests
4. Submit pull request
5. Code review and merge

### Code Standards

- PEP 8 compliance
- Type hints for all functions
- Comprehensive docstrings
- Error handling best practices
- Security-first development

## 📞 Support

### Issues and Bug Reports

- Use GitHub Issues for bug reports
- Include reproduction steps
- Provide environment details
- Attach relevant logs

### Feature Requests

- Describe use case and benefits
- Provide implementation suggestions
- Consider security implications
- Discuss with maintainers

## 📄 License

This project is part of the Pixelated Empathy platform and follows the
organization's licensing terms. See the main project repository for license
details.

## 🙏 Acknowledgments

- Pixelated Empathy development team
- Open source contributors
- Mental health professionals providing domain expertise
- Security auditors and compliance experts

---

For more information about the Pixelated Empathy platform, visit the
[main project documentation](https://github.com/pixelated-empathy/pixelated).
