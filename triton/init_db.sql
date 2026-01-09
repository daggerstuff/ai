-- Initialize PostgreSQL database for Pixel inference metadata and results

-- Create schema
CREATE SCHEMA IF NOT EXISTS pixel_inference;
SET search_path TO pixel_inference;

-- Models table
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB
);

CREATE INDEX idx_models_name ON models(name);
CREATE INDEX idx_models_version ON models(version);
CREATE INDEX idx_models_active ON models(is_active);

-- Inference sessions table
CREATE TABLE IF NOT EXISTS inference_sessions (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL UNIQUE,
    model_id INTEGER REFERENCES models(id),
    user_id VARCHAR(255),
    context_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50),
    metadata JSONB
);

CREATE INDEX idx_sessions_session_id ON inference_sessions(session_id);
CREATE INDEX idx_sessions_user_id ON inference_sessions(user_id);
CREATE INDEX idx_sessions_created_at ON inference_sessions(created_at);
CREATE INDEX idx_sessions_status ON inference_sessions(status);

-- Inference requests table
CREATE TABLE IF NOT EXISTS inference_requests (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES inference_sessions(session_id),
    request_id VARCHAR(255) UNIQUE,
    input_text TEXT,
    input_length INTEGER,
    batch_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    latency_ms FLOAT,
    model_version VARCHAR(50),
    metadata JSONB
);

CREATE INDEX idx_requests_session_id ON inference_requests(session_id);
CREATE INDEX idx_requests_request_id ON inference_requests(request_id);
CREATE INDEX idx_requests_created_at ON inference_requests(created_at);

-- Inference results table
CREATE TABLE IF NOT EXISTS inference_results (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) UNIQUE REFERENCES inference_requests(request_id),
    response_text TEXT,
    eq_scores FLOAT8[],  -- Array of 5 EQ domain scores
    overall_eq FLOAT,
    bias_score FLOAT,
    safety_score FLOAT,
    persona_mode VARCHAR(100),
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_results_request_id ON inference_results(request_id);
CREATE INDEX idx_results_overall_eq ON inference_results(overall_eq);
CREATE INDEX idx_results_bias_score ON inference_results(bias_score);
CREATE INDEX idx_results_safety_score ON inference_results(safety_score);

-- Inference metrics table
CREATE TABLE IF NOT EXISTS inference_metrics (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    request_count INTEGER,
    error_count INTEGER,
    avg_latency_ms FLOAT,
    p50_latency_ms FLOAT,
    p99_latency_ms FLOAT,
    throughput_rps FLOAT,
    gpu_utilization FLOAT,
    gpu_memory_usage_mb INTEGER,
    batch_size_avg FLOAT,
    cache_hit_rate FLOAT,
    metadata JSONB
);

CREATE INDEX idx_metrics_model_id ON inference_metrics(model_id);
CREATE INDEX idx_metrics_timestamp ON inference_metrics(timestamp);

-- A/B test configurations table
CREATE TABLE IF NOT EXISTS ab_tests (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(255) NOT NULL UNIQUE,
    control_model_id INTEGER REFERENCES models(id),
    treatment_model_id INTEGER REFERENCES models(id),
    control_traffic_percent FLOAT DEFAULT 50.0,
    treatment_traffic_percent FLOAT DEFAULT 50.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50),
    metadata JSONB
);

CREATE INDEX idx_ab_tests_status ON ab_tests(status);
CREATE INDEX idx_ab_tests_created_at ON ab_tests(created_at);

-- A/B test results table
CREATE TABLE IF NOT EXISTS ab_test_results (
    id SERIAL PRIMARY KEY,
    test_id INTEGER REFERENCES ab_tests(id),
    variant VARCHAR(50),  -- 'control' or 'treatment'
    session_id UUID REFERENCES inference_sessions(session_id),
    metric_name VARCHAR(255),
    metric_value FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_ab_results_test_id ON ab_test_results(test_id);
CREATE INDEX idx_ab_results_variant ON ab_test_results(variant);
CREATE INDEX idx_ab_results_metric_name ON ab_test_results(metric_name);

-- Crisis alerts table
CREATE TABLE IF NOT EXISTS crisis_alerts (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES inference_sessions(session_id),
    risk_level VARCHAR(50),  -- 'low', 'medium', 'high', 'critical'
    risk_score FLOAT,
    trigger_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolution VARCHAR(255),
    metadata JSONB
);

CREATE INDEX idx_crisis_alerts_session_id ON crisis_alerts(session_id);
CREATE INDEX idx_crisis_alerts_risk_level ON crisis_alerts(risk_level);
CREATE INDEX idx_crisis_alerts_created_at ON crisis_alerts(created_at);

-- Bias incidents table
CREATE TABLE IF NOT EXISTS bias_incidents (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES inference_sessions(session_id),
    bias_type VARCHAR(100),
    bias_score FLOAT,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    reviewed_by VARCHAR(255),
    action_taken VARCHAR(255),
    metadata JSONB
);

CREATE INDEX idx_bias_incidents_session_id ON bias_incidents(session_id);
CREATE INDEX idx_bias_incidents_bias_type ON bias_incidents(bias_type);
CREATE INDEX idx_bias_incidents_created_at ON bias_incidents(created_at);

-- Performance alerts table
CREATE TABLE IF NOT EXISTS performance_alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(100),
    alert_level VARCHAR(50),  -- 'warning', 'critical'
    message TEXT,
    threshold_value FLOAT,
    actual_value FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

CREATE INDEX idx_perf_alerts_alert_type ON performance_alerts(alert_type);
CREATE INDEX idx_perf_alerts_level ON performance_alerts(alert_level);
CREATE INDEX idx_perf_alerts_created_at ON performance_alerts(created_at);

-- Create views for common queries

-- View: Recent inference accuracy
CREATE OR REPLACE VIEW recent_inference_accuracy AS
SELECT
    m.name,
    m.version,
    COUNT(ir.id) as total_requests,
    AVG(ir.latency_ms) as avg_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ir.latency_ms) as p99_latency_ms,
    AVG(ires.overall_eq) as avg_eq_score,
    AVG(ires.safety_score) as avg_safety_score,
    AVG(ires.bias_score) as avg_bias_score
FROM models m
LEFT JOIN inference_requests ir ON m.id = (SELECT model_id FROM inference_sessions WHERE session_id = ir.session_id LIMIT 1)
LEFT JOIN inference_results ires ON ir.request_id = ires.request_id
WHERE ir.created_at > NOW() - INTERVAL '24 hours'
GROUP BY m.id, m.name, m.version;

-- View: Model comparison for A/B tests
CREATE OR REPLACE VIEW ab_test_comparison AS
SELECT
    t.test_name,
    t.status,
    vc.variant,
    COUNT(*) as request_count,
    AVG(ires.overall_eq) as avg_eq_score,
    AVG(ires.bias_score) as avg_bias_score,
    AVG(ires.safety_score) as avg_safety_score,
    AVG(ir.latency_ms) as avg_latency_ms
FROM ab_tests t
LEFT JOIN ab_test_results vc ON t.id = vc.test_id
LEFT JOIN inference_sessions ises ON vc.session_id = ises.session_id
LEFT JOIN inference_requests ir ON ises.session_id = ir.session_id
LEFT JOIN inference_results ires ON ir.request_id = ires.request_id
GROUP BY t.test_name, t.status, vc.variant;

-- Grant permissions
GRANT USAGE ON SCHEMA pixel_inference TO pixel_user;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pixel_inference TO pixel_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA pixel_inference TO pixel_user;
