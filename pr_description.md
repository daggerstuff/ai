🎯 **What:** The user requested to remove an alleged unused import `from .routes import router as search_router` in `sourcing/academic/api/main.py`. However, static analysis (via AST parsing and Ruff) confirmed that `search_router` is actually used in `app.include_router(search_router, prefix="/api")`. No changes were necessary as the codebase was already healthy in this regard. This PR contains an empty commit to submit the task.

💡 **Why:** The codebase is already healthy. Removing the import would have broken the API routes. A verification script and AST checks confirmed it.

✅ **Verification:**
- Ran a local python script using `ast` which confirmed `search_router` is used.
- Created a functional FastAPI `TestClient` script with mocked dependencies, verifying `/health` successfully responds with HTTP 200.
- Ran the full `pytest` suite ensuring no regressions were introduced.

✨ **Result:** Confirmed the code health is already optimal for this file.
