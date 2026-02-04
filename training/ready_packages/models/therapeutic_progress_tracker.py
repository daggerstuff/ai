#!/usr/bin/env python3
"""
Therapeutic Progress Tracking System
Long-term, journal-style tracking of client therapeutic progress
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import statistics


class EmotionalState(Enum):
    """Client emotional states"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class ProgressTrajectory(Enum):
    """Overall progress trajectory"""
    IMPROVING = "improving"
    STABLE = "stable"
    REGRESSING = "regressing"
    FLUCTUATING = "fluctuating"


@dataclass
class TherapeuticGoal:
    """Individual therapeutic goal"""
    goal_id: str
    description: str
    target_date: Optional[datetime] = None
    completion_percentage: float = 0.0
    milestones: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SessionLog:
    """Individual therapy session log"""
    session_id: str
    client_id: str
    timestamp: datetime
    conversation_summary: str
    emotional_state: EmotionalState
    therapeutic_goals: List[str]  # Goal IDs
    progress_notes: str
    therapist_observations: str
    next_session_focus: str
    session_duration_minutes: int = 60
    techniques_used: List[str] = field(default_factory=list)
    homework_assigned: str = ""
    crisis_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmotionalTrend:
    """Emotional state trend over time"""
    start_date: datetime
    end_date: datetime
    avg_emotional_score: float  # -2 to +2
    trend_direction: str  # "improving", "stable", "declining"
    volatility: float  # Standard deviation
    data_points: int


@dataclass
class Milestone:
    """Therapeutic milestone"""
    milestone_id: str
    goal_id: str
    description: str
    achieved_date: Optional[datetime] = None
    significance: str = ""  # "minor", "moderate", "major"


@dataclass
class ProgressReport:
    """Comprehensive progress report"""
    client_id: str
    report_date: datetime
    timeframe_start: datetime
    timeframe_end: datetime
    sessions_count: int
    goal_progress: Dict[str, float]  # goal_id -> completion %
    emotional_trends: List[EmotionalTrend]
    key_milestones: List[Milestone]
    overall_trajectory: ProgressTrajectory
    recommendations: List[str]
    summary: str


class TherapeuticProgressTracker:
    """
    Long-term therapeutic progress tracking system
    Supports journal-style logging and extended timeframe analysis
    """
    
    def __init__(self, db_path: str = "therapeutic_progress.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                conversation_summary TEXT,
                emotional_state TEXT,
                therapeutic_goals TEXT,
                progress_notes TEXT,
                therapist_observations TEXT,
                next_session_focus TEXT,
                session_duration_minutes INTEGER,
                techniques_used TEXT,
                homework_assigned TEXT,
                crisis_flags TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Goals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS goals (
                goal_id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                description TEXT NOT NULL,
                target_date TEXT,
                completion_percentage REAL DEFAULT 0.0,
                milestones TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Milestones table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS milestones (
                milestone_id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL,
                client_id TEXT NOT NULL,
                description TEXT NOT NULL,
                achieved_date TEXT,
                significance TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_client ON sessions(client_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_goals_client ON goals(client_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_milestones_client ON milestones(client_id)')
        
        conn.commit()
        conn.close()
    
    def log_session(self, session: SessionLog):
        """Log a therapy session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (
                session_id, client_id, timestamp, conversation_summary,
                emotional_state, therapeutic_goals, progress_notes,
                therapist_observations, next_session_focus, session_duration_minutes,
                techniques_used, homework_assigned, crisis_flags, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session.session_id,
            session.client_id,
            session.timestamp.isoformat(),
            session.conversation_summary,
            session.emotional_state.value,
            json.dumps(session.therapeutic_goals),
            session.progress_notes,
            session.therapist_observations,
            session.next_session_focus,
            session.session_duration_minutes,
            json.dumps(session.techniques_used),
            session.homework_assigned,
            json.dumps(session.crisis_flags),
            json.dumps(session.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def create_goal(self, client_id: str, goal: TherapeuticGoal):
        """Create a therapeutic goal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO goals (
                goal_id, client_id, description, target_date,
                completion_percentage, milestones, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            goal.goal_id,
            client_id,
            goal.description,
            goal.target_date.isoformat() if goal.target_date else None,
            goal.completion_percentage,
            json.dumps(goal.milestones),
            goal.notes
        ))
        
        conn.commit()
        conn.close()
    
    def update_goal_progress(self, goal_id: str, completion_percentage: float, notes: str = ""):
        """Update goal progress"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE goals
            SET completion_percentage = ?,
                notes = ?,
                updated_at = ?
            WHERE goal_id = ?
        ''', (completion_percentage, notes, datetime.now().isoformat(), goal_id))
        
        conn.commit()
        conn.close()
    
    def add_milestone(self, milestone: Milestone, client_id: str):
        """Add a milestone achievement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO milestones (
                milestone_id, goal_id, client_id, description,
                achieved_date, significance
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            milestone.milestone_id,
            milestone.goal_id,
            client_id,
            milestone.description,
            milestone.achieved_date.isoformat() if milestone.achieved_date else None,
            milestone.significance
        ))
        
        conn.commit()
        conn.close()
    
    def get_sessions(
        self,
        client_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[SessionLog]:
        """Retrieve sessions for a client"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM sessions WHERE client_id = ?'
        params = [client_id]
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date.isoformat())
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date.isoformat())
        
        query += ' ORDER BY timestamp DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            sessions.append(SessionLog(
                session_id=row[0],
                client_id=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                conversation_summary=row[3],
                emotional_state=EmotionalState(row[4]),
                therapeutic_goals=json.loads(row[5]),
                progress_notes=row[6],
                therapist_observations=row[7],
                next_session_focus=row[8],
                session_duration_minutes=row[9],
                techniques_used=json.loads(row[10]),
                homework_assigned=row[11],
                crisis_flags=json.loads(row[12]),
                metadata=json.loads(row[13])
            ))
        
        return sessions
    
    def get_goals(self, client_id: str, active_only: bool = True) -> List[TherapeuticGoal]:
        """Get therapeutic goals for a client"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM goals WHERE client_id = ?'
        params = [client_id]
        
        if active_only:
            query += ' AND completion_percentage < 100.0'
        
        query += ' ORDER BY created_at DESC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        goals = []
        for row in rows:
            goals.append(TherapeuticGoal(
                goal_id=row[0],
                description=row[2],
                target_date=datetime.fromisoformat(row[3]) if row[3] else None,
                completion_percentage=row[4],
                milestones=json.loads(row[5]),
                notes=row[6],
                created_at=datetime.fromisoformat(row[7]),
                updated_at=datetime.fromisoformat(row[8])
            ))
        
        return goals
    
    def analyze_emotional_trends(
        self,
        client_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EmotionalTrend]:
        """Analyze emotional trends over time"""
        sessions = self.get_sessions(client_id, start_date, end_date)
        
        if not sessions:
            return []
        
        # Convert emotional states to scores
        emotion_scores = {
            EmotionalState.VERY_NEGATIVE: -2,
            EmotionalState.NEGATIVE: -1,
            EmotionalState.NEUTRAL: 0,
            EmotionalState.POSITIVE: 1,
            EmotionalState.VERY_POSITIVE: 2
        }
        
        scores = [emotion_scores[s.emotional_state] for s in sessions]
        
        # Calculate statistics
        avg_score = statistics.mean(scores)
        volatility = statistics.stdev(scores) if len(scores) > 1 else 0.0
        
        # Determine trend direction
        if len(scores) >= 3:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            if second_avg > first_avg + 0.3:
                trend_direction = "improving"
            elif second_avg < first_avg - 0.3:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "insufficient_data"
        
        return [EmotionalTrend(
            start_date=start_date,
            end_date=end_date,
            avg_emotional_score=avg_score,
            trend_direction=trend_direction,
            volatility=volatility,
            data_points=len(scores)
        )]
    
    def generate_progress_report(
        self,
        client_id: str,
        timeframe_days: int = 30
    ) -> ProgressReport:
        """Generate comprehensive progress report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=timeframe_days)
        
        # Get sessions
        sessions = self.get_sessions(client_id, start_date, end_date)
        
        # Get goals
        goals = self.get_goals(client_id, active_only=False)
        goal_progress = {g.goal_id: g.completion_percentage for g in goals}
        
        # Analyze emotional trends
        emotional_trends = self.analyze_emotional_trends(client_id, start_date, end_date)
        
        # Get milestones
        milestones = self._get_milestones(client_id, start_date, end_date)
        
        # Determine overall trajectory
        trajectory = self._determine_trajectory(sessions, goals, emotional_trends)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(sessions, goals, emotional_trends)
        
        # Generate summary
        summary = self._generate_summary(sessions, goals, emotional_trends, trajectory)
        
        return ProgressReport(
            client_id=client_id,
            report_date=datetime.now(),
            timeframe_start=start_date,
            timeframe_end=end_date,
            sessions_count=len(sessions),
            goal_progress=goal_progress,
            emotional_trends=emotional_trends,
            key_milestones=milestones,
            overall_trajectory=trajectory,
            recommendations=recommendations,
            summary=summary
        )
    
    def _get_milestones(
        self,
        client_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Milestone]:
        """Get milestones achieved in timeframe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM milestones
            WHERE client_id = ?
            AND achieved_date >= ?
            AND achieved_date <= ?
            ORDER BY achieved_date DESC
        ''', (client_id, start_date.isoformat(), end_date.isoformat()))
        
        rows = cursor.fetchall()
        conn.close()
        
        milestones = []
        for row in rows:
            milestones.append(Milestone(
                milestone_id=row[0],
                goal_id=row[1],
                description=row[3],
                achieved_date=datetime.fromisoformat(row[4]) if row[4] else None,
                significance=row[5]
            ))
        
        return milestones
    
    def _determine_trajectory(
        self,
        sessions: List[SessionLog],
        goals: List[TherapeuticGoal],
        emotional_trends: List[EmotionalTrend]
    ) -> ProgressTrajectory:
        """Determine overall progress trajectory"""
        if not sessions or not emotional_trends:
            return ProgressTrajectory.STABLE
        
        # Check emotional trend
        trend = emotional_trends[0]
        
        # Check goal progress
        active_goals = [g for g in goals if g.completion_percentage < 100.0]
        if active_goals:
            avg_goal_progress = statistics.mean([g.completion_percentage for g in active_goals])
        else:
            avg_goal_progress = 100.0
        
        # Determine trajectory
        if trend.trend_direction == "improving" and avg_goal_progress > 50:
            return ProgressTrajectory.IMPROVING
        elif trend.trend_direction == "declining":
            return ProgressTrajectory.REGRESSING
        elif trend.volatility > 1.0:
            return ProgressTrajectory.FLUCTUATING
        else:
            return ProgressTrajectory.STABLE
    
    def _generate_recommendations(
        self,
        sessions: List[SessionLog],
        goals: List[TherapeuticGoal],
        emotional_trends: List[EmotionalTrend]
    ) -> List[str]:
        """Generate recommendations based on progress"""
        recommendations = []
        
        if not sessions:
            recommendations.append("Schedule initial assessment session")
            return recommendations
        
        # Check session frequency
        if len(sessions) < 4:
            recommendations.append("Consider increasing session frequency")
        
        # Check emotional trends
        if emotional_trends and emotional_trends[0].trend_direction == "declining":
            recommendations.append("Review and adjust therapeutic approach")
            recommendations.append("Consider crisis intervention protocols")
        
        # Check goal progress
        stalled_goals = [g for g in goals if g.completion_percentage < 20 and 
                        (datetime.now() - g.created_at).days > 30]
        if stalled_goals:
            recommendations.append("Re-evaluate stalled therapeutic goals")
        
        # Check techniques
        recent_sessions = sessions[:5]
        techniques = set()
        for s in recent_sessions:
            techniques.update(s.techniques_used)
        
        if len(techniques) < 2:
            recommendations.append("Consider diversifying therapeutic techniques")
        
        return recommendations
    
    def _generate_summary(
        self,
        sessions: List[SessionLog],
        goals: List[TherapeuticGoal],
        emotional_trends: List[EmotionalTrend],
        trajectory: ProgressTrajectory
    ) -> str:
        """Generate progress summary"""
        if not sessions:
            return "No sessions recorded in this timeframe."
        
        summary_parts = []
        
        # Session summary
        summary_parts.append(f"Completed {len(sessions)} sessions in this period.")
        
        # Emotional summary
        if emotional_trends:
            trend = emotional_trends[0]
            summary_parts.append(
                f"Emotional state trending {trend.trend_direction} "
                f"with average score of {trend.avg_emotional_score:.2f}."
            )
        
        # Goal summary
        active_goals = [g for g in goals if g.completion_percentage < 100.0]
        completed_goals = [g for g in goals if g.completion_percentage >= 100.0]
        
        if completed_goals:
            summary_parts.append(f"Achieved {len(completed_goals)} therapeutic goals.")
        
        if active_goals:
            avg_progress = statistics.mean([g.completion_percentage for g in active_goals])
            summary_parts.append(
                f"Active goals showing {avg_progress:.1f}% average progress."
            )
        
        # Overall trajectory
        summary_parts.append(f"Overall trajectory: {trajectory.value}.")
        
        return " ".join(summary_parts)
    
    def export_client_history(
        self,
        client_id: str,
        output_path: str,
        format: str = "json"
    ):
        """Export complete client history"""
        sessions = self.get_sessions(client_id)
        goals = self.get_goals(client_id, active_only=False)
        
        data = {
            'client_id': client_id,
            'export_date': datetime.now().isoformat(),
            'sessions': [asdict(s) for s in sessions],
            'goals': [asdict(g) for g in goals]
        }
        
        # Convert datetime objects to strings
        def convert_datetimes(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetimes(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetimes(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            return obj
        
        data = convert_datetimes(data)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Example usage
    print("🧠 Therapeutic Progress Tracking System")
    print("=" * 60)
    
    tracker = TherapeuticProgressTracker()
    
    # Example: Log a session
    session = SessionLog(
        session_id="session_001",
        client_id="client_123",
        timestamp=datetime.now(),
        conversation_summary="Discussed anxiety management techniques",
        emotional_state=EmotionalState.NEUTRAL,
        therapeutic_goals=["goal_001"],
        progress_notes="Client showing improved coping skills",
        therapist_observations="Engaged and receptive",
        next_session_focus="Practice mindfulness exercises"
    )
    
    tracker.log_session(session)
    print("✅ Session logged")
    
    # Example: Create a goal
    goal = TherapeuticGoal(
        goal_id="goal_001",
        description="Reduce anxiety symptoms",
        target_date=datetime.now() + timedelta(days=90),
        completion_percentage=25.0
    )
    
    tracker.create_goal("client_123", goal)
    print("✅ Goal created")
    
    # Example: Generate progress report
    report = tracker.generate_progress_report("client_123", timeframe_days=30)
    print(f"\n📊 Progress Report:")
    print(f"   Sessions: {report.sessions_count}")
    print(f"   Trajectory: {report.overall_trajectory.value}")
    print(f"   Summary: {report.summary}")
