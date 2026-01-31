"""
Web dashboard for Pixel Voice pipeline management.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import structlog

from ..api.auth import get_current_user, User, auth_manager, UserRole
from ..api.database import get_db, JobRepository, UsageRepository
from ..api.models import JobStatus, PipelineStage

logger = structlog.get_logger(__name__)

# Templates and static files
templates = Jinja2Templates(directory="pixel_voice/web/templates")


def setup_dashboard(app: FastAPI):
    """Setup web dashboard routes."""

    # Mount static files
    app.mount("/static", StaticFiles(directory="pixel_voice/web/static"), name="static")

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_home(
        request: Request, current_user: User = Depends(get_current_user), db=Depends(get_db)
    ):
        """Dashboard home page."""
        job_repo = JobRepository(db)
        usage_repo = UsageRepository(db)

        # Get user's recent jobs
        recent_jobs = job_repo.list_user_jobs(current_user.id, limit=10)

        # Get usage statistics
        today = datetime.now()
        daily_usage = usage_repo.get_daily_usage(current_user.id, "api_call", today)

        # Get job statistics
        job_stats = {
            "total": len(recent_jobs),
            "completed": len([j for j in recent_jobs if j.status == JobStatus.COMPLETED.value]),
            "running": len([j for j in recent_jobs if j.status == JobStatus.RUNNING.value]),
            "failed": len([j for j in recent_jobs if j.status == JobStatus.FAILED.value]),
        }

        return templates.TemplateResponse(
            "dashboard/home.html",
            {
                "request": request,
                "user": current_user,
                "recent_jobs": recent_jobs,
                "job_stats": job_stats,
                "daily_usage": daily_usage,
            },
        )

    @app.get("/dashboard/jobs", response_class=HTMLResponse)
    async def dashboard_jobs(
        request: Request,
        status: Optional[str] = None,
        page: int = 1,
        current_user: User = Depends(get_current_user),
        db=Depends(get_db),
    ):
        """Jobs management page."""
        job_repo = JobRepository(db)

        # Pagination
        limit = 20
        skip = (page - 1) * limit

        jobs = job_repo.list_user_jobs(current_user.id, status=status, skip=skip, limit=limit)

        return templates.TemplateResponse(
            "dashboard/jobs.html",
            {
                "request": request,
                "user": current_user,
                "jobs": jobs,
                "current_status": status,
                "current_page": page,
                "job_statuses": [s.value for s in JobStatus],
                "pipeline_stages": [s.value for s in PipelineStage],
            },
        )

    @app.get("/dashboard/jobs/{job_id}", response_class=HTMLResponse)
    async def dashboard_job_detail(
        request: Request,
        job_id: str,
        current_user: User = Depends(get_current_user),
        db=Depends(get_db),
    ):
        """Job detail page."""
        job_repo = JobRepository(db)

        job = job_repo.get_job_by_id(job_id)
        if not job or job.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Job not found")

        return templates.TemplateResponse(
            "dashboard/job_detail.html", {"request": request, "user": current_user, "job": job}
        )

    @app.get("/dashboard/create-job", response_class=HTMLResponse)
    async def dashboard_create_job_form(
        request: Request, current_user: User = Depends(get_current_user)
    ):
        """Create job form page."""
        return templates.TemplateResponse(
            "dashboard/create_job.html",
            {
                "request": request,
                "user": current_user,
                "pipeline_stages": [s.value for s in PipelineStage],
            },
        )

    @app.post("/dashboard/create-job")
    async def dashboard_create_job(
        request: Request,
        job_name: str = Form(...),
        youtube_url: str = Form(None),
        stages: List[str] = Form(...),
        current_user: User = Depends(get_current_user),
    ):
        """Handle job creation from dashboard."""
        try:
            # Import here to avoid circular imports
            from ..api.server import pipeline_executor

            input_data = {}
            if youtube_url:
                input_data["youtube_url"] = youtube_url

            job_id = await pipeline_executor.execute_pipeline_job(
                job_name=job_name,
                stages=[PipelineStage(stage) for stage in stages],
                input_data=input_data,
            )

            return RedirectResponse(url=f"/dashboard/jobs/{job_id}", status_code=303)

        except Exception as e:
            logger.error("Job creation failed", error=str(e), user_id=current_user.id)
            return templates.TemplateResponse(
                "dashboard/create_job.html",
                {
                    "request": request,
                    "user": current_user,
                    "pipeline_stages": [s.value for s in PipelineStage],
                    "error": str(e),
                },
            )

    @app.get("/dashboard/usage", response_class=HTMLResponse)
    async def dashboard_usage(
        request: Request, current_user: User = Depends(get_current_user), db=Depends(get_db)
    ):
        """Usage statistics page."""
        usage_repo = UsageRepository(db)

        # Get usage data for the last 30 days
        usage_data = []
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            api_calls = usage_repo.get_daily_usage(current_user.id, "api_call", date)
            youtube_downloads = usage_repo.get_daily_usage(
                current_user.id, "youtube_download", date
            )

            usage_data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "api_calls": api_calls,
                    "youtube_downloads": youtube_downloads,
                }
            )

        usage_data.reverse()  # Show oldest first

        return templates.TemplateResponse(
            "dashboard/usage.html",
            {"request": request, "user": current_user, "usage_data": usage_data},
        )

    @app.get("/dashboard/settings", response_class=HTMLResponse)
    async def dashboard_settings(request: Request, current_user: User = Depends(get_current_user)):
        """User settings page."""
        return templates.TemplateResponse(
            "dashboard/settings.html", {"request": request, "user": current_user}
        )

    @app.post("/dashboard/settings")
    async def dashboard_update_settings(
        request: Request,
        username: str = Form(None),
        email: str = Form(None),
        current_user: User = Depends(get_current_user),
    ):
        """Update user settings."""
        try:
            update_data = {}
            if username and username != current_user.username:
                update_data["username"] = username
            if email and email != current_user.email:
                update_data["email"] = email

            if update_data:
                auth_manager.update_user(current_user.id, update_data)

            return RedirectResponse(url="/dashboard/settings", status_code=303)

        except Exception as e:
            logger.error("Settings update failed", error=str(e), user_id=current_user.id)
            return templates.TemplateResponse(
                "dashboard/settings.html",
                {"request": request, "user": current_user, "error": str(e)},
            )

    @app.get("/dashboard/admin", response_class=HTMLResponse)
    async def dashboard_admin(
        request: Request, current_user: User = Depends(get_current_user), db=Depends(get_db)
    ):
        """Admin dashboard (admin users only)."""
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Admin access required")

        # Get system statistics
        job_repo = JobRepository(db)

        # This would need to be implemented in the repository
        # system_stats = job_repo.get_system_statistics()

        return templates.TemplateResponse(
            "dashboard/admin.html",
            {
                "request": request,
                "user": current_user,
                # "system_stats": system_stats
            },
        )

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request):
        """Login form page."""
        return templates.TemplateResponse("auth/login.html", {"request": request})

    @app.post("/login")
    async def login(request: Request, email: str = Form(...), password: str = Form(...)):
        """Handle login."""
        try:
            # This would need to be implemented in auth_manager
            # user = auth_manager.authenticate_user(email, password)
            # if user:
            #     # Create session or redirect with token
            #     return RedirectResponse(url="/dashboard", status_code=303)

            return templates.TemplateResponse(
                "auth/login.html", {"request": request, "error": "Invalid credentials"}
            )

        except Exception as e:
            logger.error("Login failed", error=str(e), email=email)
            return templates.TemplateResponse(
                "auth/login.html", {"request": request, "error": "Login failed"}
            )

    @app.get("/register", response_class=HTMLResponse)
    async def register_form(request: Request):
        """Registration form page."""
        return templates.TemplateResponse("auth/register.html", {"request": request})

    @app.post("/register")
    async def register(
        request: Request,
        username: str = Form(...),
        email: str = Form(...),
        password: str = Form(...),
    ):
        """Handle registration."""
        try:
            # This would need to be implemented in auth_manager
            # user = auth_manager.create_user(UserCreate(
            #     username=username,
            #     email=email,
            #     password=password
            # ))

            return RedirectResponse(url="/login", status_code=303)

        except Exception as e:
            logger.error("Registration failed", error=str(e), email=email)
            return templates.TemplateResponse(
                "auth/register.html", {"request": request, "error": "Registration failed"}
            )

    logger.info("Web dashboard setup completed")
