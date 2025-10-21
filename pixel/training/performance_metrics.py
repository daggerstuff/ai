"""
Performance Metrics (Tier 1.15)

Model-free utilities to compute latency/throughput/error metrics from request logs
or in-process measurements. Provides Prometheus and JSON exporters.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import time
import json


@dataclass
class PerfEvent:
    ts: float
    latency_ms: float
    status: int = 200
    route: str = ""
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerfAggregator:
    def __init__(self) -> None:
        self.events: List[PerfEvent] = []
        self._last_window_start: float = time.time()

    def record(self, latency_ms: float, status: int = 200, route: str = "", model: str = "") -> None:
        self.events.append(PerfEvent(ts=time.time(), latency_ms=float(latency_ms), status=int(status), route=route, model=model))

    def import_jsonl(self, path: str) -> int:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                ts = float(o.get("ts", time.time()))
                lat = float(o.get("latency_ms", o.get("latency", 0)))
                status = int(o.get("status", 200))
                route = o.get("route", o.get("path", ""))
                model = o.get("model", "")
                self.events.append(PerfEvent(ts=ts, latency_ms=lat, status=status, route=route, model=model))
                count += 1
        return count

    def _percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = (len(values)-1) * p
        f = int(k)
        c = min(f+1, len(values)-1)
        if f == c:
            return values[int(k)]
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return d0 + d1

    def _window(self, since: Optional[float] = None) -> List[PerfEvent]:
        if since is None:
            return self.events
        return [e for e in self.events if e.ts >= since]

    def metrics(self, since: Optional[float] = None) -> Dict[str, Any]:
        ev = self._window(since)
        latencies = [e.latency_ms for e in ev]
        count = len(ev)
        errors = sum(1 for e in ev if e.status >= 500)
        client_errors = sum(1 for e in ev if 400 <= e.status < 500)
        throughput = count / max(1.0, (time.time() - (since or (self._last_window_start))))
        return {
            "count": count,
            "throughput_rps": throughput,
            "latency": {
                "p50": self._percentile(latencies, 0.5),
                "p90": self._percentile(latencies, 0.9),
                "p95": self._percentile(latencies, 0.95),
                "p99": self._percentile(latencies, 0.99),
                "avg": sum(latencies)/len(latencies) if latencies else 0.0,
            },
            "errors": {
                "5xx": errors,
                "4xx": client_errors,
                "rate": (errors + client_errors) / count if count else 0.0,
            },
        }

    def export_prometheus(self, since: Optional[float] = None) -> str:
        m = self.metrics(since)
        lines = []
        lines.append(f"pixel_perf_throughput_rps {m['throughput_rps']}")
        lines.append(f"pixel_perf_count {m['count']}")
        lines.append(f"pixel_perf_latency_p50_ms {m['latency']['p50']}")
        lines.append(f"pixel_perf_latency_p90_ms {m['latency']['p90']}")
        lines.append(f"pixel_perf_latency_p95_ms {m['latency']['p95']}")
        lines.append(f"pixel_perf_latency_p99_ms {m['latency']['p99']}")
        lines.append(f"pixel_perf_latency_avg_ms {m['latency']['avg']}")
        lines.append(f"pixel_perf_errors_5xx {m['errors']['5xx']}")
        lines.append(f"pixel_perf_errors_4xx {m['errors']['4xx']}")
        lines.append(f"pixel_perf_error_rate {m['errors']['rate']}")
        return "\n".join(lines) + "\n"

    def export_json(self, since: Optional[float] = None) -> str:
        return json.dumps(self.metrics(since), ensure_ascii=False)
