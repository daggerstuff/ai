#!/usr/bin/env python3
"""
Progress Tracking API
RESTful API for therapeutic progress tracking
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

from models.therapeutic_progress_tracker import (
    TherapeuticProgressTracker,
    SessionLog,
    TherapeuticGoal,
    Milestone,
    EmotionalState,
    ProgressTrajectory
)


# Pydantic models for API
class EmotionalStateModel(str, Enum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class SessionLogRequest(BaseModel):
    session_id: str
    client_id: str
    conversation_summary: str
    emotional_state: EmotionalStateModel
    therapeutic_goals: List[str]
    progress_notes: str
    therapist_observations: str
    next_session_focus: str
    session_duration_minutes: int = 60
    techniques_used: List[str] = []
    homework_assigned: str = ""
    crisis_flags: List[str] = []
    metadata: Dict[str, Any] = {}


class GoalRequest(BaseModel):
    goal_id: str
    client_id: str
    description: str
    target_date: Optional[datetime] = None
    completion_percentage: float = 0.0
    milestones: List[str] = []
    notes: str = ""


class GoalUpdateRequest(BaseModel):
    completion_percentage: float
    notes: str = ""


class MilestoneRequest(BaseModel):
    milestone_id: str
    goal_id: str
    client_id: str
    description: str
    achieved_date: Optional[datetime] = None
    significance: str = "moderate"


class ProgressReportResponse(BaseModel):
    client_id: str
    report_date: datetime
    timeframe_start: datetime
    timeframe_end: datetime
    sessions_count: int
    goal_progress: Dict[str, float]
    overall_trajectory: str
    recommendations: List[str]
    summary: str


# Create FastAPI app
app = FastAPI(
    title="Therapeutic Progress Tracking API",
    description="API for long-term therapeutic progress tracking",
    version="1.0.0"
)

# Initialize tracker
tracker = TherapeuticProgressTracker()


@app.post("/api/v1/sessions")
async def log_session(request: SessionLogRequest):
    """Log a therapy session"""
    try:
        session = SessionLog(
            session_id=request.session_id,
            client_id=request.client_id,
            timestamp=datetime.now(),
            conversation_summary=request.conversation_summary,
            emotional_state=EmotionalState(request.emotional_state.value),
            therapeutic_goals=request.therapeutic_goals,
            progress_notes=request.progress_notes,
            therapist_observations=request.therapist_observations,
            next_session_focus=request.next_session_focus,
            session_duration_minutes=request.session_duration_minutes,
            techniques_used=request.techniques_used,
            homework_assigned=request.homework_assigned,
            crisis_flags=request.crisis_flags,
            metadata=request.metadata
        )

        tracker.log_session(session)

        return {
            "status": "success",
            "message": "Session logged successfully",
            "session_id": request.session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/sessions/{client_id}")
async def get_sessions(
    client_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: Optional[int] = None
):
    """Get sessions for a client"""
    try:
        sessions = tracker.get_sessions(client_id, start_date, end_date, limit)

        return {
            "client_id": client_id,
            "sessions_count": len(sessions),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "timestamp": s.timestamp.isoformat(),
                    "emotional_state": s.emotional_state.value,
                    "conversation_summary": s.conversation_summary,
                    "progress_notes": s.progress_notes
                }
                for s in sessions
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/goals")
async def create_goal(request: GoalRequest):
    """Create a therapeutic goal"""
    try:
        goal = TherapeuticGoal(
            goal_id=request.goal_id,
            description=request.description,
            target_date=request.target_date,
            completion_percentage=request.completion_percentage,
            milestones=request.milestones,
            notes=request.notes
        )

        tracker.create_goal(request.client_id, goal)

        return {
            "status": "success",
            "message": "Goal created successfully",
            "goal_id": request.goal_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/goals/{goal_id}")
async def update_goal(goal_id: str, request: GoalUpdateRequest):
    """Update goal progress"""
    try:
        tracker.update_goal_progress(
            goal_id,
            request.completion_percentage,
            request.notes
        )

        return {
            "status": "success",
            "message": "Goal updated successfully",
            "goal_id": goal_id,
            "completion_percentage": request.completion_percentage
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/goals/{client_id}")
async def get_goals(client_id: str, active_only: bool = True):
    """Get therapeutic goals for a client"""
    try:
        goals = tracker.get_goals(client_id, active_only)

        return {
            "client_id": client_id,
            "goals_count": len(goals),
            "goals": [
                {
                    "goal_id": g.goal_id,
                    "description": g.description,
                    "completion_percentage": g.completion_percentage,
                    "target_date": g.target_date.isoformat() if g.target_date else None,
                    "created_at": g.created_at.isoformat()
                }
                for g in goals
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/milestones")
async def add_milestone(request: MilestoneRequest):
    """Add a milestone achievement"""
    try:
        milestone = Milestone(
            milestone_id=request.milestone_id,
            goal_id=request.goal_id,
            description=request.description,
            achieved_date=request.achieved_date or datetime.now(),
            significance=request.significance
        )

        tracker.add_milestone(milestone, request.client_id)

        return {
            "status": "success",
            "message": "Milestone added successfully",
            "milestone_id": request.milestone_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/progress/{client_id}", response_model=ProgressReportResponse)
async def get_progress_report(client_id: str, timeframe_days: int = 30):
    """Generate progress report for a client"""
    try:
        report = tracker.generate_progress_report(client_id, timeframe_days)

        return ProgressReportResponse(
            client_id=report.client_id,
            report_date=report.report_date,
            timeframe_start=report.timeframe_start,
            timeframe_end=report.timeframe_end,
            sessions_count=report.sessions_count,
            goal_progress=report.goal_progress,
            overall_trajectory=report.overall_trajectory.value,
            recommendations=report.recommendations,
            summary=report.summary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/trends/{client_id}")
async def get_emotional_trends(
    client_id: str,
    timeframe_days: int = 30
):
    """Get emotional trends for a client"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=timeframe_days)

        trends = tracker.analyze_emotional_trends(client_id, start_date, end_date)

        return {
            "client_id": client_id,
            "timeframe_days": timeframe_days,
            "trends": [
                {
                    "avg_emotional_score": t.avg_emotional_score,
                    "trend_direction": t.trend_direction,
                    "volatility": t.volatility,
                    "data_points": t.data_points
                }
                for t in trends
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/export/{client_id}")
async def export_history(client_id: str):
    """Export complete client history"""
    try:
        output_path = f"client_{client_id}_history.json"
        tracker.export_client_history(client_id, output_path)

        return {
            "status": "success",
            "message": "History exported successfully",
            "file_path": output_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Progress Tracking API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
