"""Interactive web dashboards over the quality metrics database.

Serves the executive quality view as a live Flask app: one page with the band
distribution, the daily quality trend, the per-tier means and the outstanding
anomalies, plus JSON endpoints for the same data. Every number comes from the
``conversations``/``quality_metrics`` tables; when the database is absent the
endpoints say so instead of inventing figures.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify

from .quality_analytics_dashboard import QualityAnalytics, QualityAnalyticsDashboard

DEFAULT_PAGE_TITLE = "Quality analytics"
CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"

_SERIES_PALETTE = ["#55d6e0", "#e8e6df", "#9a9aa3", "#7f7f88", "#3a3a42"]


class InteractiveDashboardSystem:
    """A Flask app plus the data assembly behind its pages."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        app: Flask | None = None,
    ) -> None:
        self.db_path = db_path if db_path is None or isinstance(db_path, str) else str(db_path)
        self.app = app or Flask(__name__)
        self.dashboard = QualityAnalyticsDashboard(db_path=self.db_path)
        self._routes_registered = False

    def _analytics(self) -> tuple[QualityAnalytics, bool]:
        if self.db_path is None or not Path(self.db_path).exists():
            return QualityAnalytics(), False
        frame = self.dashboard.load_quality_data(force_refresh=True)
        return self.dashboard.calculate_quality_analytics(frame), True

    def get_dashboard_data(self) -> dict[str, Any]:
        """Assemble the payload the page and the JSON endpoints render."""
        analytics, available = self._analytics()
        trend_labels = [point["date"] for point in analytics.trend_data]
        trend_values = [round(point["quality"], 4) for point in analytics.trend_data]
        bands = analytics.quality_distribution
        anomalies = analytics.anomalies

        return {
            "available": available,
            "metrics": {
                "total_conversations": analytics.total_conversations,
                "average_quality": round(analytics.average_quality, 4),
                "anomaly_count": len(anomalies),
                "tier_count": len(analytics.tier_performance),
            },
            "quality_distribution": dict(bands),
            "tier_performance": {tier: round(score, 4) for tier, score in analytics.tier_performance.items()},
            "trend_labels": trend_labels,
            "trend_values": trend_values,
            "anomalies": anomalies,
            "recommendations": list(analytics.recommendations),
            "current_time": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    def render_executive_page(self) -> str:
        """Render the executive dashboard as a self-contained HTML page."""
        data = self.get_dashboard_data()
        payload = json.dumps(
            {
                "bands": data["quality_distribution"],
                "trend_labels": data["trend_labels"],
                "trend_values": data["trend_values"],
                "tiers": data["tier_performance"],
                "palette": _SERIES_PALETTE,
            }
        )
        recommendations = (
            "".join(f"<li>{item}</li>" for item in data["recommendations"]) or "<li>No recommendations this cycle.</li>"
        )
        anomaly_rows = (
            "".join(
                f"<tr><td>{anomaly['id']}</td><td>{anomaly['tier']}</td>"
                f"<td>{anomaly['dataset']}</td><td>{anomaly['quality']:.3f}</td>"
                f"<td>{anomaly['direction']}</td></tr>"
                for anomaly in data["anomalies"]
            )
            or '<tr><td colspan="5">No conversations outside the interquartile fence.</td></tr>'
        )
        status = (
            f"{data['metrics']['total_conversations']} conversations, mean {data['metrics']['average_quality']:.3f}"
            if data["available"]
            else "No quality database is configured, so there is nothing to chart."
        )

        return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{DEFAULT_PAGE_TITLE}</title>
<script src="{CHART_JS_CDN}"></script>
<style>
:root {{
  --np-bg: #0b0b0c; --np-surface: #121214; --np-elevated: #18181b;
  --np-text: #e8e6df; --np-muted: #8b8b94; --np-line: #1c1c21; --np-line-2: #2a2a30;
  --np-cyan: #55d6e0; --np-amber: #ffb000;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 32px; background: var(--np-bg); color: var(--np-text);
  font-family: "Instrument Sans", system-ui, sans-serif; }}
h1 {{ font-size: 1.5rem; font-weight: 700; margin: 0 0 4px; }}
p.lede {{ color: var(--np-muted); margin: 0 0 24px; font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.8125rem; letter-spacing: 0.04em; text-transform: uppercase; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
section {{ background: var(--np-surface); border: 1px solid var(--np-line); padding: 24px; }}
section h2 {{ font-size: 0.75rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--np-muted); font-family: "IBM Plex Mono", ui-monospace, monospace; margin: 0 0 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
th {{ text-align: left; color: var(--np-muted); font-weight: 500; border-bottom: 1px solid var(--np-line-2);
  padding: 8px 4px; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.6875rem;
  text-transform: uppercase; letter-spacing: 0.14em; }}
td {{ padding: 8px 4px; border-bottom: 1px solid var(--np-line); font-variant-numeric: tabular-nums; }}
ul {{ margin: 0; padding-left: 20px; line-height: 1.6; }}
canvas {{ width: 100% !important; }}
</style>
</head>
<body>
<h1>{DEFAULT_PAGE_TITLE}</h1>
<p class="lede">{status} &middot; generated {data["current_time"]}</p>
<div class="grid">
  <section><h2>Band distribution</h2><canvas id="bands" height="200"></canvas></section>
  <section><h2>Daily mean quality</h2><canvas id="trend" height="200"></canvas></section>
  <section><h2>Mean quality by tier</h2><canvas id="tiers" height="200"></canvas></section>
</div>
<div class="grid" style="margin-top:16px">
  <section>
    <h2>Anomalies</h2>
    <table><thead><tr><th>ID</th><th>Tier</th><th>Dataset</th><th>Quality</th><th>Direction</th></tr></thead>
    <tbody>{anomaly_rows}</tbody></table>
  </section>
  <section><h2>Recommendations</h2><ul>{recommendations}</ul></section>
</div>
<script>
const data = {payload};
const chartDefaults = {{
  responsive: true,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    x: {{ ticks: {{ color: "#8b8b94" }}, grid: {{ color: "#1c1c21" }} }},
    y: {{ ticks: {{ color: "#8b8b94" }}, grid: {{ color: "#1c1c21" }} }}
  }}
}};
new Chart(document.getElementById("bands"), {{
  type: "bar",
  data: {{ labels: Object.keys(data.bands), datasets: [{{
    data: Object.values(data.bands),
    backgroundColor: data.palette.slice(0, Object.keys(data.bands).length)
  }}] }},
  options: chartDefaults
}});
new Chart(document.getElementById("trend"), {{
  type: "line",
  data: {{ labels: data.trend_labels, datasets: [{{
    data: data.trend_values, borderColor: data.palette[0], backgroundColor: data.palette[0],
    tension: 0.2, pointRadius: 2
  }}] }},
  options: Object.assign({{}}, chartDefaults, {{ scales: Object.assign({{}}, chartDefaults.scales,
    {{ y: {{ min: 0, max: 1, ticks: {{ color: "#8b8b94" }}, grid: {{ color: "#1c1c21" }} }} }}) }})
}});
new Chart(document.getElementById("tiers"), {{
  type: "bar",
  data: {{ labels: Object.keys(data.tiers), datasets: [{{
    data: Object.values(data.tiers),
    backgroundColor: data.palette.slice(0, Object.keys(data.tiers).length)
  }}] }},
  options: Object.assign({{}}, chartDefaults, {{ scales: Object.assign({{}}, chartDefaults.scales,
    {{ y: {{ min: 0, max: 1, ticks: {{ color: "#8b8b94" }}, grid: {{ color: "#1c1c21" }} }} }}) }})
}});
</script>
</body>
</html>
"""

    def setup_flask_routes(self) -> Flask:
        """Register the dashboard routes once and return the app."""
        if self._routes_registered:
            return self.app

        @self.app.route("/")
        @self.app.route("/executive")
        def executive_dashboard() -> Response:
            return Response(self.render_executive_page(), mimetype="text/html")

        @self.app.route("/api/executive-metrics")
        def api_executive_metrics() -> Response:
            data = self.get_dashboard_data()
            return jsonify(
                {
                    "metrics": data["metrics"],
                    "quality_distribution": data["quality_distribution"],
                    "tier_performance": data["tier_performance"],
                    "available": data["available"],
                }
            )

        @self.app.route("/api/quality-trend")
        def api_quality_trend() -> Response:
            data = self.get_dashboard_data()
            return jsonify({"labels": data["trend_labels"], "values": data["trend_values"]})

        @self.app.route("/api/anomalies")
        def api_anomalies() -> Response:
            return jsonify({"anomalies": self.get_dashboard_data()["anomalies"]})

        @self.app.route("/api/health")
        def api_health() -> Response:
            exists = self.db_path is not None and Path(self.db_path).exists()
            return jsonify(
                {
                    "status": "ok" if exists else "degraded",
                    "database": self.db_path,
                    "database_present": exists,
                    "checked_at": datetime.now(UTC).isoformat(),
                }
            )

        self._routes_registered = True
        return self.app

    def deploy_interactive_dashboards(self, output_dir: str | Path | None = None) -> Path:
        """Write the executive page to disk and return the path written."""
        target_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent / "dashboards" / "executive"
        target_dir.mkdir(parents=True, exist_ok=True)
        page = target_dir / "executive.html"
        page.write_text(self.render_executive_page(), encoding="utf-8")
        return page


def main() -> None:
    """Serve the dashboard with the database path from the environment."""
    import os

    system = InteractiveDashboardSystem(db_path=os.environ.get("QUALITY_DB_PATH"))
    system.setup_flask_routes()
    system.app.run(host=os.environ.get("DASHBOARD_HOST", "127.0.0.1"), port=5000, debug=False)


if __name__ == "__main__":
    main()
