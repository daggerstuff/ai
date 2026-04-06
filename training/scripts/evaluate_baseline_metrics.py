# Remove duplicate REPO_ROOT definition (Review suggestion)
DATASET_DIRS = [
    REPO_ROOT / "ai/data/acquired_datasets",
    REPO_ROOT / "ai/training/data/generated",
    REPO_ROOT / "ai/data/raw",
]


def _scan_all_datasets(evaluator: Any) -> int:
    logger.info("Scanning all dataset directories...")
    all_results: dict[str, Any] = {}

    for scan_dir in DATASET_DIRS:
        if not scan_dir.exists():
            logger.warning(f"Directory not found: {scan_dir}")
            continue

        for file_path in sorted(scan_dir.iterdir()):
            if (
                file_path.suffix in (".json", ".jsonl")
                and "_stats" not in file_path.name
                and "_report" not in file_path.name
                and "summary" not in file_path.name
            ):
                result = print_evaluation_report(
                    evaluator, file_path, f"--- {file_path.name} ---"
                )
                all_results[file_path.name] = result

    report_path = REPO_ROOT / "ai/data/reports/phase2_baseline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Consolidated report: {report_path}")
    return 0


def _run_dry_run_evaluation(evaluator: Any) -> None:
    logger.info("Running DRY RUN evaluation...")
