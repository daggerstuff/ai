"""Simple analytics pipeline dashboard with alerting and reporting utilities."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class MetricRecord:
    """Single metric snapshot."""

    metric_type: str
    data: dict[str, Any]
    collected_at: str
    alerts_triggered: list[str] = field(default_factory=list)


@dataclass
class ReportSummary:
    """Structured summary returned by report operations."""

    report_id: str
    metric_type: str | None
    time_period: str
    created_at: str
    total_records: int
    aggregate_metrics: dict[str, float]


class AnalyticsDashboard:
    """Production-friendly metric collector with optional alerting."""

    def __init__(
        self,
        *,
        alert_thresholds: dict[str, float] | None = None,
        max_retention_days: int = 30,
    ) -> None:
        self.metrics_data: dict[str, list[MetricRecord]] = {}
        self.alert_thresholds: dict[str, float] = {
            "response_time": 1.0,
            "error_rate": 0.05,
            "memory_usage": 0.85,
            "cpu_usage": 0.90,
            "loss": 1.0,
        }
        if alert_thresholds:
            self.alert_thresholds.update(
                {k: float(v) for k, v in alert_thresholds.items() if isinstance(v, (int, float))}
            )
        self.reports_generated: list[ReportSummary] = []
        self._report_data: dict[str, dict[str, Any]] = {}
        self.monitoring_active = True
        self.max_retention_days = max_retention_days

    def collect_metrics(self, metric_type: str, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(metric_type, str) or not metric_type.strip():
            return {
                "success": False,
                "error": "metric_type must be a non-empty string",
                "metrics_collected": 0,
                "metric_type": None,
            }
        if not isinstance(data, dict) or not data:
            return {
                "success": False,
                "error": "data must be a non-empty dict",
                "metrics_collected": 0,
                "metric_type": metric_type,
            }

        alerts = self._evaluate_alerts(data)
        rec = MetricRecord(
            metric_type=metric_type,
            data=data,
            collected_at=datetime.now(UTC).isoformat(),
            alerts_triggered=alerts,
        )
        bucket = self.metrics_data.setdefault(metric_type, [])
        bucket.append(rec)
        return {
            "success": True,
            "error": None,
            "metric_type": metric_type,
            "metrics_collected": 1,
            "active_alerts": alerts,
        }

    def _evaluate_alerts(self, data: dict[str, Any]) -> list[str]:
        alerts: list[str] = []
        for key, threshold in self.alert_thresholds.items():
            if key not in data:
                continue
            value = data.get(key)
            try:
                if isinstance(value, (int, float)) and value >= threshold:
                    alerts.append(f"{key} exceeds threshold ({value} >= {threshold})")
            except Exception:
                continue
        return alerts

    def _cleanup_expired(self) -> None:
        if self.max_retention_days <= 0:
            return
        cutoff = datetime.now(UTC) - timedelta(days=self.max_retention_days)
        for metric_type, rows in list(self.metrics_data.items()):
            kept: list[MetricRecord] = []
            for row in rows:
                try:
                    ts = datetime.fromisoformat(row.collected_at)
                except Exception:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts >= cutoff:
                    kept.append(row)
            self.metrics_data[metric_type] = kept

    def generate_performance_report(self, time_period: str = "all") -> dict[str, Any]:
        self._cleanup_expired()

        all_records = [record for records in self.metrics_data.values() for record in records]
        if not all_records:
            return {"success": False, "error": "No metrics available", "report": None}

        if time_period != "all":
            all_records = self._filter_time_window(all_records, time_period)

        if not all_records:
            return {"success": False, "error": f"No metrics for period '{time_period}'", "report": None}

        aggregate = self._aggregate_numeric_fields(all_records)
        summary = ReportSummary(
            report_id=f"rpt-{uuid.uuid4().hex[:12]}",
            metric_type="all",
            time_period=time_period,
            created_at=datetime.now(UTC).isoformat(),
            total_records=len(all_records),
            aggregate_metrics=aggregate,
        )
        report_data = {
            "report_id": summary.report_id,
            "metric_type": summary.metric_type,
            "time_period": summary.time_period,
            "created_at": summary.created_at,
            "data": [r.__dict__ for r in all_records],
            "summary": summary.aggregate_metrics,
        }
        self.reports_generated.append(summary)
        self._report_data[summary.report_id] = report_data

        return {
            "success": True,
            "error": None,
            "report": report_data,
        }

    def _filter_time_window(self, rows: list[MetricRecord], time_period: str) -> list[MetricRecord]:
        now = datetime.now(UTC)
        if time_period == "last_24h":
            cutoff = now - timedelta(hours=24)
        elif time_period == "last_7d":
            cutoff = now - timedelta(days=7)
        elif time_period == "last_30d":
            cutoff = now - timedelta(days=30)
        else:
            return rows

        filtered: list[MetricRecord] = []
        for row in rows:
            try:
                ts = datetime.fromisoformat(row.collected_at)
            except Exception:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts >= cutoff:
                filtered.append(row)
        return filtered

    def _aggregate_numeric_fields(self, rows: list[MetricRecord]) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in rows:
            for key, value in row.data.items():
                if not isinstance(value, (int, float)) or math.isnan(float(value)):
                    continue
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1

        return {key: value / counts[key] for key, value in sums.items() if counts.get(key, 0) > 0}

    def export_report(self, export_format: str = "json", export_dir: str = "exports") -> dict[str, Any]:
        if not self.reports_generated:
            return {"success": False, "error": "No reports generated", "export_path": None}

        export_format = export_format.lower()
        if export_format not in {"json", "csv"}:
            return {"success": False, "error": "Unsupported export format", "export_path": None}

        latest = self.reports_generated[-1]
        report_data = self._report_data.get(latest.report_id)
        if report_data is None:
            return {"success": False, "error": "Report data not found for latest report", "export_path": None}

        export_dir_path = Path(export_dir)
        export_dir_path.mkdir(parents=True, exist_ok=True)
        export_path = export_dir_path / f"{latest.report_id}.{export_format}"

        if export_format == "json":
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, default=str)
        else:
            records = report_data.get("data", [])
            fieldnames: list[str] = []
            for rec in records:
                for key in rec.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            with open(export_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)

        return {
            "success": True,
            "error": None,
            "export_path": str(export_path),
            "export_format": export_format,
            "records_exported": latest.total_records,
        }

    def set_alert_thresholds(self, thresholds: dict[str, float]) -> dict[str, Any]:
        if not isinstance(thresholds, dict):
            return {"success": False, "error": "Invalid thresholds configuration", "updated_thresholds": None}

        for key, value in thresholds.items():
            if (key in self.alert_thresholds and isinstance(value, (int, float))) or isinstance(value, (int, float)):
                self.alert_thresholds[key] = float(value)

        return {"success": True, "error": None, "updated_thresholds": self.alert_thresholds.copy()}

    def get_dashboard_statistics(self) -> dict[str, Any]:
        total_metrics = sum(len(records) for records in self.metrics_data.values())
        return {
            "total_metrics_collected": total_metrics,
            "metric_types": len(self.metrics_data),
            "reports_generated": len(self.reports_generated),
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "alert_thresholds": self.alert_thresholds.copy(),
            "data_retention_days": self.max_retention_days,
        }


__all__ = [
    "AnalyticsDashboard",
    "MetricRecord",
    "ReportSummary",
]
