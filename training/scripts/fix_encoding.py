def save_results(
    *,
    project_root: Path,
    dry_run: bool,
    results: list[dict[str, Any]],
) -> Path:
    """Save encoding fix results to JSON file"""
    categories = categorize_results(results)
    successful = categories["successful"]
    failed = categories["failed"]
    skipped = categories["skipped"]
    fixed = categories["fixed"]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dry_run_suffix = "_dry_run" if dry_run else ""
    report_path = (
        project_root
        / "ai/data/reports"
        / f"encoding_fix_report_{timestamp}{dry_run_suffix}.json"
    )
