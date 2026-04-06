def categorize_results(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Categorizes results into successful, failed, skipped, and fixed.
    Extracted into shared helper to avoid duplication (Review suggestion).
    """
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    skipped = [r for r in successful if r.get("skipped")]
    fixed = [r for r in successful if not r.get("skipped")]
    
    return {
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "fixed": fixed
    }


def print_results(results: list[dict[str, Any]], output: OutputHandler) -> None:
    """Print summary of encoding fix results"""
    output.info("")
    output.separator()
    output.header("📊 ENCODING FIX RESULTS")
    output.separator()

    categories = categorize_results(results)
    successful = categories["successful"]
    failed = categories["failed"]
    skipped = categories["skipped"]
    fixed = categories["fixed"]
