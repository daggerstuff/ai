#!/usr/bin/env python3
"""
Therapy Assistant System Implementation
AI assistance for licensed therapists during therapeutic sessions.

This system provides real-time support to therapists with:
- Session analysis and insights
- Intervention suggestions
- Documentation assistance
- Treatment plan recommendations
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TherapyApproach(Enum):
    CBT = "cognitive_behavioral_therapy"
    DBT = "dialectical_behavior_therapy"
    EMDR = "eye_movement_desensitization_reprocessing"
    PSYCHODYNAMIC = "psychodynamic"
    HUMANISTIC = "humanistic"
    SOLUTION_FOCUSED = "solution_focused"
    TRAUMA_INFORMED = "trauma_informed"
    FAMILY_SYSTEMS = "family_systems"

class SessionPhase(Enum):
    OPENING = "opening"
    EXPLORATION = "exploration"
    INTERVENTION = "intervention"
    PROCESSING = "processing"
    CLOSING = "closing"

class InterventionType(Enum):
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    EMOTIONAL_REGULATION = "emotional_regulation"
    BEHAVIORAL_ACTIVATION = "behavioral_activation"
    MINDFULNESS = "mindfulness"
    GROUNDING = "grounding"
    PSYCHOEDUCATION = "psychoeducation"
    HOMEWORK_ASSIGNMENT = "homework_assignment"
    SAFETY_PLANNING = "safety_planning"

@dataclass
class SessionAnalysis:
    """Real-time session analysis"""
    session_id: str
    current_phase: SessionPhase
    primary_themes: List[str]
    emotional_state: Dict[str, float]  # emotion -> intensity
    therapeutic_alliance: float  # 0-1 scale
    progress_indicators: List[str]
    concerns: List[str]
    suggested_interventions: List[Dict]
    analysis_timestamp: datetime

@dataclass
class InterventionSuggestion:
    """Therapeutic intervention suggestion"""
    intervention_type: InterventionType
    description: str
    rationale: str
    timing: str  # "immediate", "later_in_session", "next_session"
    therapy_approach: TherapyApproach
    expected_outcome: str
    considerations: List[str]
    confidence: float

@dataclass
class SessionDocumentation:
    """Automated session documentation assistance"""
    session_summary: str
    key_topics_discussed: List[str]
    interventions_used: List[str]
    client_insights: List[str]
    homework_assigned: List[str]
    progress_notes: str
    treatment_plan_updates: List[str]
    next_session_focus: List[str]
    risk_assessment: Dict
    documentation_timestamp: datetime

class SessionAnalyzer:
    """Analyze therapeutic sessions in real-time"""
    
    def __init__(self):
        # Therapeutic themes detection patterns
        self.theme_patterns = {
            "anxiety": [
                r"worried|anxious|nervous|panic|fear|scared|stress",
                r"can't stop thinking|racing thoughts|overthinking",
                r"what if|worst case|catastrophizing"
            ],
            "depression": [
                r"sad|depressed|hopeless|empty|numb|down",
                r"no energy|tired|exhausted|can't get up",
                r"worthless|failure|hate myself|no point"
            ],
            "trauma": [
                r"flashback|nightmare|triggered|memories",
                r"happened to me|abuse|assault|accident",
                r"can't forget|keeps happening|reliving"
            ],
            "relationships": [
                r"my partner|my spouse|relationship|marriage",
                r"family|parents|children|friends",
                r"conflict|argument|communication|trust"
            ],
            "self_esteem": [
                r"not good enough|inadequate|insecure",
                r"comparing myself|self-worth|confidence",
                r"criticism|judgment|approval"
            ],
            "grief": [
                r"loss|died|death|passing|funeral",
                r"missing|gone|never see again",
                r"grieving|mourning|bereaved"
            ],
            "substance_use": [
                r"drinking|alcohol|drugs|using|high",
                r"addiction|sober|recovery|relapse",
                r"can't control|need to use|withdrawal"
            ]
        }
        
        # Emotional state indicators
        self.emotion_indicators = {
            "anger": [r"angry|mad|furious|rage|pissed|irritated"],
            "sadness": [r"sad|crying|tears|heartbroken|devastated"],
            "fear": [r"afraid|scared|terrified|fearful|frightened"],
            "joy": [r"happy|joyful|excited|elated|thrilled"],
            "shame": [r"ashamed|embarrassed|humiliated|mortified"],
            "guilt": [r"guilty|fault|blame|responsible|regret"],
            "hope": [r"hopeful|optimistic|better|improve|future"],
            "confusion": [r"confused|don't understand|mixed up|unclear"]
        }
        
        # Progress indicators
        self.progress_patterns = {
            "insight": [
                r"i realize|i understand now|it makes sense",
                r"i see the pattern|i notice|i'm aware"
            ],
            "behavior_change": [
                r"i tried|i did|i practiced|i used the technique",
                r"different this time|new approach|changed my response"
            ],
            "emotional_regulation": [
                r"i stayed calm|i breathed|i grounded myself",
                r"didn't react|took a step back|managed my emotions"
            ],
            "social_connection": [
                r"talked to|reached out|connected with",
                r"opened up|shared with|asked for help"
            ]
        }
        
        # Session phase indicators
        self.phase_indicators = {
            SessionPhase.OPENING: [
                r"how are you|how was your week|what's been happening",
                r"good to see you|thanks for coming|let's start"
            ],
            SessionPhase.EXPLORATION: [
                r"tell me more|what was that like|how did you feel",
                r"when did this start|what triggered|can you describe"
            ],
            SessionPhase.INTERVENTION: [
                r"let's try|practice this|technique|exercise",
                r"what if we|different way|alternative|strategy"
            ],
            SessionPhase.PROCESSING: [
                r"what came up|how was that|what did you notice",
                r"reaction to|thoughts about|insights|realizations"
            ],
            SessionPhase.CLOSING: [
                r"wrapping up|almost time|few minutes left",
                r"next week|homework|between sessions|see you"
            ]
        }
    
    def analyze_session_content(self, transcript_segment: str, session_context: Dict) -> SessionAnalysis:
        """Analyze a segment of session transcript"""
        segment_lower = transcript_segment.lower()
        
        # Detect current session phase
        current_phase = self._detect_session_phase(segment_lower, session_context)
        
        # Identify primary themes
        primary_themes = self._identify_themes(segment_lower)
        
        # Assess emotional state
        emotional_state = self._assess_emotional_state(segment_lower)
        
        # Evaluate therapeutic alliance
        therapeutic_alliance = self._evaluate_alliance(segment_lower, session_context)
        
        # Identify progress indicators
        progress_indicators = self._identify_progress(segment_lower)
        
        # Detect concerns
        concerns = self._detect_concerns(segment_lower, emotional_state)
        
        # Generate intervention suggestions
        suggested_interventions = self._suggest_interventions(
            primary_themes, emotional_state, current_phase, session_context
        )
        
        return SessionAnalysis(
            session_id=session_context.get("session_id", "unknown"),
            current_phase=current_phase,
            primary_themes=primary_themes,
            emotional_state=emotional_state,
            therapeutic_alliance=therapeutic_alliance,
            progress_indicators=progress_indicators,
            concerns=concerns,
            suggested_interventions=suggested_interventions,
            analysis_timestamp=datetime.now()
        )
    
    def _detect_session_phase(self, text: str, context: Dict) -> SessionPhase:
        """Detect current phase of therapy session"""
        phase_scores = {}
        
        for phase, patterns in self.phase_indicators.items():
            score = 0
            for pattern in patterns:
                score += len(re.findall(pattern, text))
            phase_scores[phase] = score
        
        # Consider session timing
        session_duration = context.get("session_duration_minutes", 0)
        if session_duration < 10:
            return SessionPhase.OPENING
        elif session_duration > 45:
            return SessionPhase.CLOSING
        
        # Return phase with highest score, default to exploration
        return max(phase_scores.items(), key=lambda x: x[1])[0] if any(phase_scores.values()) else SessionPhase.EXPLORATION
    
    def _identify_themes(self, text: str) -> List[str]:
        """Identify primary therapeutic themes"""
        themes = []
        
        for theme, patterns in self.theme_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    themes.append(theme)
                    break
        
        return themes
    
    def _assess_emotional_state(self, text: str) -> Dict[str, float]:
        """Assess client's emotional state"""
        emotions = {}
        
        for emotion, patterns in self.emotion_indicators.items():
            intensity = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text))
                intensity += matches * 0.2  # Scale matches to intensity
            
            if intensity > 0:
                emotions[emotion] = min(intensity, 1.0)
        
        return emotions
    
    def _evaluate_alliance(self, text: str, context: Dict) -> float:
        """Evaluate therapeutic alliance strength"""
        alliance_score = 0.5  # Baseline
        
        # Positive alliance indicators
        positive_indicators = [
            r"you understand|you get it|helps to talk",
            r"feel comfortable|feel safe|trust you",
            r"good session|helpful|makes sense"
        ]
        
        # Negative alliance indicators
        negative_indicators = [
            r"don't understand|doesn't help|not working",
            r"uncomfortable|judging me|don't get it",
            r"waste of time|not helping|frustrated"
        ]
        
        for pattern in positive_indicators:
            alliance_score += len(re.findall(pattern, text)) * 0.1
        
        for pattern in negative_indicators:
            alliance_score -= len(re.findall(pattern, text)) * 0.1
        
        return max(0.0, min(1.0, alliance_score))
    
    def _identify_progress(self, text: str) -> List[str]:
        """Identify progress indicators"""
        progress = []
        
        for indicator_type, patterns in self.progress_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    progress.append(indicator_type)
                    break
        
        return progress
    
    def _detect_concerns(self, text: str, emotions: Dict[str, float]) -> List[str]:
        """Detect concerning indicators"""
        concerns = []
        
        # High-intensity negative emotions
        if emotions.get("anger", 0) > 0.7:
            concerns.append("High anger intensity - monitor for aggression risk")
        if emotions.get("sadness", 0) > 0.8:
            concerns.append("Severe sadness - assess for depression/suicide risk")
        if emotions.get("fear", 0) > 0.7:
            concerns.append("High fear/anxiety - may need grounding techniques")
        
        # Crisis indicators
        crisis_patterns = [
            r"hurt myself|kill myself|end it all|not worth living",
            r"everyone would be better off|can't go on|no point",
            r"plan to|thinking about dying|suicide"
        ]
        
        for pattern in crisis_patterns:
            if re.search(pattern, text):
                concerns.append("CRISIS ALERT: Suicide risk detected - immediate assessment needed")
                break
        
        # Substance use concerns
        if re.search(r"been drinking|used drugs|high|drunk", text):
            concerns.append("Substance use mentioned - assess current state and safety")
        
        return concerns
    
    def _suggest_interventions(self, themes: List[str], emotions: Dict[str, float], 
                             phase: SessionPhase, context: Dict) -> List[Dict]:
        """Generate intervention suggestions"""
        suggestions = []
        
        # Theme-based interventions
        if "anxiety" in themes:
            suggestions.append({
                "intervention": InterventionType.GROUNDING,
                "description": "Grounding technique to manage anxiety",
                "timing": "immediate" if emotions.get("fear", 0) > 0.6 else "later_in_session"
            })
            
        if "depression" in themes:
            suggestions.append({
                "intervention": InterventionType.BEHAVIORAL_ACTIVATION,
                "description": "Behavioral activation to combat depression",
                "timing": "next_session"
            })
            
        if "trauma" in themes:
            suggestions.append({
                "intervention": InterventionType.SAFETY_PLANNING,
                "description": "Safety planning for trauma responses",
                "timing": "immediate"
            })
        
        # Emotion-based interventions
        if emotions.get("anger", 0) > 0.6:
            suggestions.append({
                "intervention": InterventionType.EMOTIONAL_REGULATION,
                "description": "Anger management and emotional regulation",
                "timing": "immediate"
            })
        
        # Phase-appropriate interventions
        if phase == SessionPhase.INTERVENTION:
            if "cognitive_restructuring" not in [s.get("intervention") for s in suggestions]:
                suggestions.append({
                    "intervention": InterventionType.COGNITIVE_RESTRUCTURING,
                    "description": "Challenge negative thought patterns",
                    "timing": "immediate"
                })
        
        return suggestions

class TherapyAssistantEngine:
    """Main therapy assistant engine"""
    
    def __init__(self, therapeutic_ai_model=None):
        self.analyzer = SessionAnalyzer()
        self.therapeutic_ai = therapeutic_ai_model
        self.active_sessions = {}
        
        # Intervention library
        self.intervention_library = {
            InterventionType.GROUNDING: {
                "techniques": [
                    "5-4-3-2-1 sensory grounding (5 things you see, 4 you hear, etc.)",
                    "Deep breathing with extended exhale",
                    "Progressive muscle relaxation",
                    "Feet on floor, hands on lap awareness"
                ],
                "when_to_use": "High anxiety, panic, dissociation",
                "considerations": "Check if client can focus on instructions"
            },
            InterventionType.COGNITIVE_RESTRUCTURING: {
                "techniques": [
                    "Thought record - identify automatic thoughts",
                    "Evidence for/against the thought",
                    "Alternative perspective generation",
                    "Socratic questioning"
                ],
                "when_to_use": "Negative thought patterns, cognitive distortions",
                "considerations": "Ensure therapeutic alliance before challenging thoughts"
            },
            InterventionType.EMOTIONAL_REGULATION: {
                "techniques": [
                    "TIPP skills (Temperature, Intense exercise, Paced breathing, Paired muscle relaxation)",
                    "Emotion identification and labeling",
                    "Distress tolerance skills",
                    "Window of tolerance education"
                ],
                "when_to_use": "Emotional overwhelm, dysregulation",
                "considerations": "Start with basic skills, build complexity gradually"
            }
        }
    
    async def provide_session_assistance(self, session_id: str, transcript_segment: str, 
                                       therapist_request: str = None) -> Dict:
        """Provide real-time assistance during therapy session"""
        
        # Get or create session context
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "start_time": datetime.now(),
                "transcript_history": [],
                "interventions_used": [],
                "session_notes": []
            }
        
        session_context = self.active_sessions[session_id]
        session_context["transcript_history"].append(transcript_segment)
        
        # Calculate session duration
        duration = (datetime.now() - session_context["start_time"]).total_seconds() / 60
        session_context["session_duration_minutes"] = duration
        
        # Analyze current segment
        analysis = self.analyzer.analyze_session_content(transcript_segment, session_context)
        
        # Generate AI-powered insights if available
        ai_insights = await self._generate_ai_insights(transcript_segment, analysis, session_context)
        
        # Create intervention recommendations
        intervention_recommendations = self._create_intervention_recommendations(analysis)
        
        # Generate documentation assistance
        documentation_assistance = self._generate_documentation_assistance(analysis, session_context)
        
        # Check for urgent concerns
        urgent_alerts = self._check_urgent_concerns(analysis)
        
        return {
            "session_analysis": {
                "current_phase": analysis.current_phase.value,
                "primary_themes": analysis.primary_themes,
                "emotional_state": analysis.emotional_state,
                "therapeutic_alliance": analysis.therapeutic_alliance,
                "progress_indicators": analysis.progress_indicators
            },
            "ai_insights": ai_insights,
            "intervention_recommendations": intervention_recommendations,
            "documentation_assistance": documentation_assistance,
            "urgent_alerts": urgent_alerts,
            "session_metadata": {
                "session_duration": duration,
                "total_segments_analyzed": len(session_context["transcript_history"]),
                "analysis_timestamp": analysis.analysis_timestamp.isoformat()
            }
        }
    
    async def _generate_ai_insights(self, transcript: str, analysis: SessionAnalysis, 
                                  context: Dict) -> Dict:
        """Generate AI-powered therapeutic insights"""
        if self.therapeutic_ai:
            # In production, this would use the trained H100 model
            # Request insights using therapeutic + educational experts
            ai_response = {
                "therapeutic_observations": "AI-generated observations about client patterns and progress",
                "suggested_questions": ["AI-suggested therapeutic questions based on content"],
                "client_strengths": ["AI-identified client strengths and resources"],
                "treatment_recommendations": "AI recommendations for treatment approach adjustments"
            }
        else:
            # Fallback insights based on analysis
            ai_response = {
                "therapeutic_observations": f"Client presenting with {', '.join(analysis.primary_themes)} themes",
                "suggested_questions": self._generate_suggested_questions(analysis),
                "client_strengths": self._identify_client_strengths(transcript),
                "treatment_recommendations": self._generate_treatment_recommendations(analysis)
            }
        
        return ai_response
    
    def _create_intervention_recommendations(self, analysis: SessionAnalysis) -> List[Dict]:
        """Create detailed intervention recommendations"""
        recommendations = []
        
        for suggestion in analysis.suggested_interventions:
            intervention_type = suggestion.get("intervention")
            if intervention_type in self.intervention_library:
                library_info = self.intervention_library[intervention_type]
                
                recommendation = {
                    "intervention_type": intervention_type.value,
                    "description": suggestion.get("description"),
                    "timing": suggestion.get("timing"),
                    "specific_techniques": library_info["techniques"],
                    "rationale": library_info["when_to_use"],
                    "considerations": library_info["considerations"],
                    "confidence": 0.8  # Based on analysis quality
                }
                recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_documentation_assistance(self, analysis: SessionAnalysis, context: Dict) -> Dict:
        """Generate session documentation assistance"""
        return {
            "session_summary_draft": f"Client presented with {', '.join(analysis.primary_themes)} themes. "
                                   f"Primary emotions: {', '.join(analysis.emotional_state.keys())}. "
                                   f"Therapeutic alliance: {analysis.therapeutic_alliance:.1f}/1.0.",
            "progress_notes": analysis.progress_indicators,
            "interventions_suggested": [s.get("description", "") for s in analysis.suggested_interventions],
            "next_session_focus": self._suggest_next_session_focus(analysis),
            "risk_assessment": {
                "suicide_risk": "low" if not any("CRISIS" in c for c in analysis.concerns) else "elevated",
                "substance_use": "assess" if any("substance" in c.lower() for c in analysis.concerns) else "none_reported",
                "safety_concerns": analysis.concerns
            }
        }
    
    def _check_urgent_concerns(self, analysis: SessionAnalysis) -> List[Dict]:
        """Check for urgent concerns requiring immediate attention"""
        alerts = []
        
        for concern in analysis.concerns:
            if "CRISIS" in concern:
                alerts.append({
                    "severity": "critical",
                    "type": "suicide_risk",
                    "message": concern,
                    "recommended_action": "Immediate suicide risk assessment and safety planning"
                })
            elif "substance" in concern.lower():
                alerts.append({
                    "severity": "high",
                    "type": "substance_use",
                    "message": concern,
                    "recommended_action": "Assess current intoxication level and session safety"
                })
        
        return alerts
    
    def _generate_suggested_questions(self, analysis: SessionAnalysis) -> List[str]:
        """Generate suggested therapeutic questions"""
        questions = []
        
        if "anxiety" in analysis.primary_themes:
            questions.append("What does the anxiety feel like in your body right now?")
            questions.append("When you notice the anxiety starting, what thoughts go through your mind?")
        
        if "depression" in analysis.primary_themes:
            questions.append("What has been giving you any sense of meaning or purpose lately?")
            questions.append("How has your energy and motivation been this week?")
        
        if analysis.progress_indicators:
            questions.append("What do you think helped you make that positive change?")
            questions.append("How can we build on this progress you're making?")
        
        return questions
    
    def _identify_client_strengths(self, transcript: str) -> List[str]:
        """Identify client strengths from transcript"""
        strengths = []
        
        strength_patterns = {
            "Self-awareness": r"i realize|i notice|i'm aware|i understand",
            "Resilience": r"got through|survived|kept going|didn't give up",
            "Social support": r"talked to|friend|family|support|reached out",
            "Coping skills": r"breathing|meditation|exercise|music|journal",
            "Motivation": r"want to change|willing to try|ready to work|committed"
        }
        
        for strength, pattern in strength_patterns.items():
            if re.search(pattern, transcript.lower()):
                strengths.append(strength)
        
        return strengths
    
    def _generate_treatment_recommendations(self, analysis: SessionAnalysis) -> str:
        """Generate treatment approach recommendations"""
        recommendations = []
        
        if "anxiety" in analysis.primary_themes:
            recommendations.append("Consider CBT techniques for anxiety management")
        if "trauma" in analysis.primary_themes:
            recommendations.append("Trauma-informed approach with grounding and safety focus")
        if analysis.therapeutic_alliance < 0.6:
            recommendations.append("Focus on strengthening therapeutic alliance before interventions")
        
        return "; ".join(recommendations) if recommendations else "Continue current therapeutic approach"
    
    def _suggest_next_session_focus(self, analysis: SessionAnalysis) -> List[str]:
        """Suggest focus areas for next session"""
        focus_areas = []
        
        if analysis.concerns:
            focus_areas.append("Address safety concerns and crisis planning")
        if "insight" in analysis.progress_indicators:
            focus_areas.append("Build on insights gained in this session")
        if analysis.therapeutic_alliance > 0.8 and analysis.primary_themes:
            focus_areas.append(f"Deeper exploration of {analysis.primary_themes[0]} theme")
        
        return focus_areas

def main():
    """Demonstrate therapy assistant system"""
    logger.info("👩‍⚕️ Therapy Assistant System Demo")
    
    # Initialize system
    assistant = TherapyAssistantEngine()
    
    # Simulate therapy session segments
    session_segments = [
        "Client: I've been having a really hard week. The anxiety has been overwhelming and I can't seem to get control of my thoughts.",
        "Therapist: That sounds really difficult. Can you tell me more about what the anxiety feels like for you?",
        "Client: It's like my heart is racing all the time and I keep thinking about all the things that could go wrong. I had a panic attack yesterday at work.",
        "Therapist: I'm sorry you experienced that panic attack. What was happening right before it started?",
        "Client: My boss wanted to meet with me and immediately I thought I was going to get fired. I know it's probably not rational, but I couldn't stop the thoughts."
    ]
    
    async def run_demo():
        session_id = "demo_session_001"
        
        for i, segment in enumerate(session_segments):
            logger.info(f"\n--- Session Segment {i+1} ---")
            logger.info(f"Content: '{segment}'")
            
            assistance = await assistant.provide_session_assistance(session_id, segment)
            
            logger.info(f"Phase: {assistance['session_analysis']['current_phase']}")
            logger.info(f"Themes: {assistance['session_analysis']['primary_themes']}")
            logger.info(f"Emotions: {assistance['session_analysis']['emotional_state']}")
            
            if assistance['intervention_recommendations']:
                logger.info(f"Suggested Intervention: {assistance['intervention_recommendations'][0]['intervention_type']}")
            
            if assistance['urgent_alerts']:
                logger.warning(f"ALERT: {assistance['urgent_alerts'][0]['message']}")
    
    asyncio.run(run_demo())
    
    logger.info("\n🎯 Therapy Assistant System ready for integration with therapeutic AI!")

if __name__ == "__main__":
    main()