# Full System Validation Report
**Generated**: 2026-05-23T10:44:57.053534+00:00
**Duration**: 8.8 seconds

## Summary
- **Total Operations**: 17
- **Successful**: 12
- **Failed**: 5
- **Success Rate**: 70.6%
- **Conversations Processed**: 0

## Performance Metrics
- **db_query_1_ms**: 100.28
- **db_query_2_ms**: 100.23
- **db_query_3_ms**: 100.24
- **export_jsonl_ms**: 500.45
- **export_parquet_ms**: 500.98
- **export_csv_ms**: 500.63
- **export_huggingface_ms**: 501.33
- **workflow_dataset_discovery_and_query_ms**: 1001.14
- **workflow_quality_validation_workflow_ms**: 1000.82
- **workflow_export_and_download_workflow_ms**: 1001.13
- **workflow_monitoring_and_analytics_workflow_ms**: 1001.85
- **stress_test_duration_seconds**: 0.10
- **stress_test_success_rate**: 100.00

## Resource Usage
- **db_test**: CPU 45.7%, Memory 15.9%
- **stress_test**: CPU 70.9%, Memory 16.2%

## Errors (5)
- 2026-05-23 10:44:48.220592+00:00: API service check failed: All connection attempts failed
- 2026-05-23 10:44:49.533481+00:00: API GET /v1/datasets failed: All connection attempts failed
- 2026-05-23 10:44:49.635069+00:00: API GET /v1/conversations failed: All connection attempts failed
- 2026-05-23 10:44:49.736623+00:00: API GET /v1/quality/metrics failed: All connection attempts failed
- 2026-05-23 10:44:49.838195+00:00: API GET /v1/monitoring/usage failed: All connection attempts failed

## Warnings (2)
- 2026-05-23 10:44:48.222904+00:00: Limited test data available: 0 (target: 4,200,000)
- 2026-05-23 10:44:57.052852+00:00: Monitoring configuration not found

## Conclusion
❌ **System validation FAILED** - Critical issues must be resolved
