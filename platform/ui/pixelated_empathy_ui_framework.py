#!/usr/bin/env python3
"""
Pixelated Empathy UI/UX Framework
Complete user interface design for the therapeutic training simulation platform.

This creates the web-based interface for:
- Trainee simulation experience
- Supervisor evaluation dashboard
- Session management and analytics
- Training program administration
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class UIComponent:
    """Base UI component definition"""
    component_id: str
    component_type: str
    title: str
    description: str
    props: Dict[str, Any]
    styling: Dict[str, str]
    interactions: List[Dict]

@dataclass
class UILayout:
    """UI layout configuration"""
    layout_id: str
    layout_type: str  # trainee, supervisor, admin
    components: List[UIComponent]
    responsive_breakpoints: Dict[str, Dict]
    theme_config: Dict[str, str]

class PixelatedEmpathyUIFramework:
    """Complete UI/UX framework for Pixelated Empathy platform"""
    
    def __init__(self):
        self.ui_components = {}
        self.layouts = {}
        self.theme_config = self._initialize_theme()
        
        # Initialize all UI components
        self._initialize_trainee_components()
        self._initialize_supervisor_components()
        self._initialize_admin_components()
        
        logger.info("🎨 Pixelated Empathy UI Framework initialized")
    
    def _initialize_theme(self) -> Dict[str, str]:
        """Initialize platform visual theme"""
        return {
            # Color Palette
            "primary_color": "#2563eb",        # Professional blue
            "secondary_color": "#7c3aed",      # Empathy purple
            "success_color": "#10b981",        # Progress green
            "warning_color": "#f59e0b",        # Caution amber
            "danger_color": "#ef4444",         # Crisis red
            "neutral_color": "#6b7280",        # Calm gray
            
            # Therapeutic Colors
            "therapeutic_blue": "#3b82f6",     # Trust and calm
            "empathy_purple": "#8b5cf6",       # Understanding
            "growth_green": "#22c55e",         # Progress and hope
            "crisis_red": "#dc2626",           # Emergency attention
            "neutral_warm": "#f3f4f6",         # Safe background
            
            # Typography
            "font_primary": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
            "font_therapeutic": "'Source Sans Pro', sans-serif",
            "font_monospace": "'JetBrains Mono', monospace",
            
            # Spacing
            "spacing_xs": "0.25rem",
            "spacing_sm": "0.5rem", 
            "spacing_md": "1rem",
            "spacing_lg": "1.5rem",
            "spacing_xl": "2rem",
            
            # Borders & Shadows
            "border_radius": "0.5rem",
            "shadow_sm": "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
            "shadow_md": "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
            "shadow_lg": "0 10px 15px -3px rgba(0, 0, 0, 0.1)"
        }
    
    def _initialize_trainee_components(self):
        """Initialize UI components for trainee interface"""
        
        # Client Information Panel
        self.ui_components["client_info_panel"] = UIComponent(
            component_id="client_info_panel",
            component_type="info_panel",
            title="Client Information",
            description="Essential client background and presentation details",
            props={
                "collapsible": True,
                "default_expanded": True,
                "sections": [
                    {
                        "title": "Demographics",
                        "fields": ["name", "age", "gender", "background"]
                    },
                    {
                        "title": "Presenting Problem", 
                        "fields": ["presenting_problem", "duration", "severity"]
                    },
                    {
                        "title": "Key Considerations",
                        "fields": ["triggers", "strengths", "therapy_goals"]
                    }
                ]
            },
            styling={
                "background": "neutral_warm",
                "border": "1px solid #e5e7eb",
                "border_radius": "border_radius",
                "padding": "spacing_lg",
                "margin_bottom": "spacing_md"
            },
            interactions=[
                {"event": "toggle_expand", "action": "toggle_panel_visibility"},
                {"event": "info_hover", "action": "show_detailed_tooltip"}
            ]
        )
        
        # Conversation Interface
        self.ui_components["conversation_interface"] = UIComponent(
            component_id="conversation_interface",
            component_type="chat_interface",
            title="Therapeutic Conversation",
            description="Main conversation interface with AI client",
            props={
                "message_history_height": "400px",
                "input_max_length": 1000,
                "typing_indicators": True,
                "emoji_support": False,  # Professional context
                "audio_cues": True,
                "client_avatar": True,
                "timestamp_display": True
            },
            styling={
                "background": "#ffffff",
                "border": "2px solid therapeutic_blue",
                "border_radius": "spacing_md",
                "min_height": "500px",
                "display": "flex",
                "flex_direction": "column"
            },
            interactions=[
                {"event": "send_message", "action": "process_therapist_input"},
                {"event": "typing_start", "action": "show_typing_indicator"},
                {"event": "message_receive", "action": "display_client_response"}
            ]
        )
        
        # Client Visual Representation
        self.ui_components["client_avatar"] = UIComponent(
            component_id="client_avatar",
            component_type="avatar_display",
            title="Client Visual Cues",
            description="Visual representation of client's emotional state and nonverbals",
            props={
                "avatar_size": "large",
                "emotion_indicators": True,
                "body_language_cues": True,
                "eye_contact_visualization": True,
                "voice_tone_indicators": True,
                "real_time_updates": True
            },
            styling={
                "width": "200px",
                "height": "200px",
                "border_radius": "50%",
                "border": "3px solid empathy_purple",
                "background": "linear-gradient(135deg, #f3f4f6, #e5e7eb)",
                "display": "flex",
                "align_items": "center",
                "justify_content": "center"
            },
            interactions=[
                {"event": "emotion_change", "action": "update_avatar_expression"},
                {"event": "hover_avatar", "action": "show_nonverbal_details"}
            ]
        )
        
        # Real-time Feedback Panel
        self.ui_components["feedback_panel"] = UIComponent(
            component_id="feedback_panel",
            component_type="feedback_display",
            title="Real-time Guidance",
            description="Live feedback and skill assessment",
            props={
                "skill_meters": [
                    "rapport_building",
                    "active_listening", 
                    "empathy",
                    "therapeutic_response"
                ],
                "progress_indicators": True,
                "suggestion_alerts": True,
                "crisis_warnings": True,
                "breakthrough_celebrations": True
            },
            styling={
                "background": "linear-gradient(to bottom, #f8fafc, #f1f5f9)",
                "border": "1px solid #cbd5e1",
                "border_radius": "border_radius",
                "padding": "spacing_lg",
                "min_height": "300px"
            },
            interactions=[
                {"event": "skill_update", "action": "animate_progress_meter"},
                {"event": "feedback_receive", "action": "highlight_new_feedback"}
            ]
        )
        
        # Session Controls
        self.ui_components["session_controls"] = UIComponent(
            component_id="session_controls",
            component_type="control_panel",
            title="Session Management",
            description="Session timing, notes, and emergency controls",
            props={
                "session_timer": True,
                "pause_resume": True,
                "emergency_stop": True,
                "note_taking": True,
                "supervisor_call": True,
                "session_summary": True
            },
            styling={
                "background": "#ffffff",
                "border": "1px solid #d1d5db",
                "border_radius": "border_radius",
                "padding": "spacing_md",
                "display": "flex",
                "justify_content": "space-between",
                "align_items": "center"
            },
            interactions=[
                {"event": "pause_session", "action": "pause_simulation"},
                {"event": "emergency_stop", "action": "trigger_emergency_protocols"},
                {"event": "call_supervisor", "action": "notify_supervisor"}
            ]
        )
    
    def _initialize_supervisor_components(self):
        """Initialize UI components for supervisor dashboard"""
        
        # Real-time Session Monitor
        self.ui_components["session_monitor"] = UIComponent(
            component_id="session_monitor",
            component_type="monitoring_dashboard",
            title="Live Session Monitoring",
            description="Real-time observation of trainee performance",
            props={
                "conversation_view": True,
                "skill_assessment_panel": True,
                "intervention_alerts": True,
                "recording_controls": True,
                "annotation_tools": True,
                "multi_session_support": True
            },
            styling={
                "background": "#ffffff",
                "border": "2px solid primary_color",
                "border_radius": "spacing_md",
                "min_height": "600px",
                "display": "grid",
                "grid_template_columns": "2fr 1fr",
                "gap": "spacing_md"
            },
            interactions=[
                {"event": "add_observation", "action": "record_supervisor_note"},
                {"event": "trigger_intervention", "action": "send_intervention_alert"},
                {"event": "update_rating", "action": "update_skill_assessment"}
            ]
        )
        
        # Competency Assessment Grid
        self.ui_components["competency_grid"] = UIComponent(
            component_id="competency_grid",
            component_type="assessment_grid",
            title="Skill Competency Assessment",
            description="Structured evaluation of therapeutic skills",
            props={
                "skill_categories": [
                    "rapport_building",
                    "active_listening",
                    "empathy_demonstration",
                    "crisis_management",
                    "resistance_handling",
                    "boundary_setting",
                    "therapeutic_confrontation",
                    "cultural_competence"
                ],
                "rating_scale": "1-5",
                "rubric_integration": True,
                "real_time_updates": True,
                "evidence_collection": True
            },
            styling={
                "background": "#f9fafb",
                "border": "1px solid #e5e7eb",
                "border_radius": "border_radius",
                "padding": "spacing_lg",
                "overflow": "auto"
            },
            interactions=[
                {"event": "rate_skill", "action": "update_competency_score"},
                {"event": "view_rubric", "action": "display_detailed_rubric"},
                {"event": "add_evidence", "action": "record_skill_evidence"}
            ]
        )
        
        # Intervention Panel
        self.ui_components["intervention_panel"] = UIComponent(
            component_id="intervention_panel",
            component_type="intervention_controls",
            title="Supervisor Interventions",
            description="Tools for providing guidance and feedback",
            props={
                "live_coaching": True,
                "pause_session": True,
                "send_suggestions": True,
                "emergency_takeover": True,
                "post_session_debrief": True,
                "development_planning": True
            },
            styling={
                "background": "linear-gradient(to right, #fef3c7, #fde68a)",
                "border": "2px solid warning_color",
                "border_radius": "border_radius",
                "padding": "spacing_lg"
            },
            interactions=[
                {"event": "send_coaching", "action": "deliver_live_feedback"},
                {"event": "schedule_intervention", "action": "plan_session_break"},
                {"event": "emergency_intervention", "action": "take_session_control"}
            ]
        )
        
        # Analytics Dashboard
        self.ui_components["analytics_dashboard"] = UIComponent(
            component_id="analytics_dashboard",
            component_type="analytics_display",
            title="Performance Analytics",
            description="Comprehensive analysis of trainee progress",
            props={
                "skill_progression_charts": True,
                "session_comparison": True,
                "cohort_benchmarking": True,
                "development_recommendations": True,
                "certification_tracking": True,
                "export_capabilities": True
            },
            styling={
                "background": "#ffffff",
                "border": "1px solid #d1d5db",
                "border_radius": "border_radius",
                "padding": "spacing_lg",
                "min_height": "400px"
            },
            interactions=[
                {"event": "filter_data", "action": "update_analytics_view"},
                {"event": "export_report", "action": "generate_progress_report"},
                {"event": "drill_down", "action": "show_detailed_analysis"}
            ]
        )
    
    def _initialize_admin_components(self):
        """Initialize UI components for administrative interface"""
        
        # Training Program Manager
        self.ui_components["program_manager"] = UIComponent(
            component_id="program_manager",
            component_type="program_administration",
            title="Training Program Management",
            description="Create and manage training curricula",
            props={
                "program_templates": True,
                "curriculum_builder": True,
                "assessment_configuration": True,
                "client_profile_library": True,
                "competency_mapping": True,
                "certification_workflows": True
            },
            styling={
                "background": "#ffffff",
                "border": "1px solid #d1d5db",
                "border_radius": "border_radius",
                "padding": "spacing_xl"
            },
            interactions=[
                {"event": "create_program", "action": "launch_program_wizard"},
                {"event": "edit_curriculum", "action": "open_curriculum_editor"},
                {"event": "configure_assessment", "action": "setup_evaluation_criteria"}
            ]
        )
        
        # User Management
        self.ui_components["user_management"] = UIComponent(
            component_id="user_management",
            component_type="user_administration",
            title="User & Role Management",
            description="Manage trainees, supervisors, and administrators",
            props={
                "role_based_access": True,
                "bulk_user_import": True,
                "progress_tracking": True,
                "permission_management": True,
                "audit_logging": True,
                "integration_apis": True
            },
            styling={
                "background": "#f8fafc",
                "border": "1px solid #e2e8f0",
                "border_radius": "border_radius",
                "padding": "spacing_lg"
            },
            interactions=[
                {"event": "add_user", "action": "create_new_user_account"},
                {"event": "assign_role", "action": "update_user_permissions"},
                {"event": "view_progress", "action": "display_user_analytics"}
            ]
        )
    
    def create_trainee_layout(self) -> UILayout:
        """Create complete layout for trainee interface"""
        
        trainee_layout = UILayout(
            layout_id="trainee_main",
            layout_type="trainee",
            components=[
                self.ui_components["client_info_panel"],
                self.ui_components["conversation_interface"],
                self.ui_components["client_avatar"],
                self.ui_components["feedback_panel"],
                self.ui_components["session_controls"]
            ],
            responsive_breakpoints={
                "mobile": {
                    "max_width": "768px",
                    "layout": "single_column",
                    "component_order": ["client_info_panel", "conversation_interface", "feedback_panel"]
                },
                "tablet": {
                    "min_width": "769px",
                    "max_width": "1024px", 
                    "layout": "two_column",
                    "main_column": ["conversation_interface", "session_controls"],
                    "side_column": ["client_info_panel", "client_avatar", "feedback_panel"]
                },
                "desktop": {
                    "min_width": "1025px",
                    "layout": "three_column",
                    "left_column": ["client_info_panel", "client_avatar"],
                    "center_column": ["conversation_interface", "session_controls"],
                    "right_column": ["feedback_panel"]
                }
            },
            theme_config=self.theme_config
        )
        
        return trainee_layout
    
    def create_supervisor_layout(self) -> UILayout:
        """Create complete layout for supervisor dashboard"""
        
        supervisor_layout = UILayout(
            layout_id="supervisor_dashboard",
            layout_type="supervisor",
            components=[
                self.ui_components["session_monitor"],
                self.ui_components["competency_grid"],
                self.ui_components["intervention_panel"],
                self.ui_components["analytics_dashboard"]
            ],
            responsive_breakpoints={
                "tablet": {
                    "min_width": "768px",
                    "max_width": "1023px",
                    "layout": "stacked",
                    "component_order": ["session_monitor", "competency_grid", "intervention_panel"]
                },
                "desktop": {
                    "min_width": "1024px",
                    "layout": "dashboard_grid",
                    "grid_areas": {
                        "session_monitor": "1 / 1 / 3 / 3",
                        "competency_grid": "1 / 3 / 2 / 4", 
                        "intervention_panel": "2 / 3 / 3 / 4",
                        "analytics_dashboard": "3 / 1 / 4 / 4"
                    }
                }
            },
            theme_config=self.theme_config
        )
        
        return supervisor_layout
    
    def generate_react_components(self) -> Dict[str, str]:
        """Generate React component code for the UI framework"""
        
        react_components = {}
        
        # Generate trainee interface components
        react_components["TraineeInterface"] = self._generate_trainee_react_code()
        react_components["SupervisorDashboard"] = self._generate_supervisor_react_code()
        react_components["ConversationInterface"] = self._generate_conversation_react_code()
        react_components["ClientAvatar"] = self._generate_avatar_react_code()
        react_components["FeedbackPanel"] = self._generate_feedback_react_code()
        
        return react_components
    
    def _generate_trainee_react_code(self) -> str:
        """Generate React code for trainee interface"""
        return '''
import React, { useState, useEffect } from 'react';
import { ConversationInterface } from './ConversationInterface';
import { ClientAvatar } from './ClientAvatar';
import { FeedbackPanel } from './FeedbackPanel';
import { ClientInfoPanel } from './ClientInfoPanel';
import { SessionControls } from './SessionControls';

export const TraineeInterface = ({ sessionId, clientProfile }) => {
  const [sessionState, setSessionState] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [realTimeFeedback, setRealTimeFeedback] = useState({});
  
  useEffect(() => {
    // Initialize session with AI client
    initializeTrainingSession(sessionId, clientProfile);
  }, [sessionId]);
  
  const handleTherapistMessage = async (message) => {
    try {
      const response = await sendTherapistMessage(sessionId, message);
      
      setConversationHistory(prev => [...prev, 
        { type: 'therapist', content: message, timestamp: new Date() },
        { type: 'client', content: response.client_response.content, timestamp: new Date() }
      ]);
      
      setRealTimeFeedback(response.real_time_feedback);
      setSessionState(response.session_status);
      
    } catch (error) {
      console.error('Error processing therapist message:', error);
    }
  };
  
  return (
    <div className="trainee-interface">
      <div className="layout-grid">
        <div className="left-column">
          <ClientInfoPanel profile={clientProfile} />
          <ClientAvatar 
            emotionalState={sessionState?.emotional_state}
            nonverbalCues={sessionState?.nonverbal_cues}
          />
        </div>
        
        <div className="center-column">
          <ConversationInterface 
            history={conversationHistory}
            onSendMessage={handleTherapistMessage}
            clientState={sessionState}
          />
          <SessionControls 
            sessionId={sessionId}
            sessionState={sessionState}
          />
        </div>
        
        <div className="right-column">
          <FeedbackPanel 
            feedback={realTimeFeedback}
            skillMetrics={sessionState?.skill_feedback}
            suggestions={sessionState?.intervention_suggestions}
          />
        </div>
      </div>
    </div>
  );
};
'''
    
    def save_ui_framework(self) -> Path:
        """Save complete UI framework to files"""
        
        ui_dir = Path("ai/platform/ui")
        ui_dir.mkdir(exist_ok=True)
        
        # Save component definitions
        components_file = ui_dir / "ui_components.json"
        with open(components_file, 'w') as f:
            json.dump({
                "components": {k: asdict(v) for k, v in self.ui_components.items()},
                "theme": self.theme_config
            }, f, indent=2)
        
        # Save layout configurations
        layouts_file = ui_dir / "ui_layouts.json"
        trainee_layout = self.create_trainee_layout()
        supervisor_layout = self.create_supervisor_layout()
        
        with open(layouts_file, 'w') as f:
            json.dump({
                "layouts": {
                    "trainee": asdict(trainee_layout),
                    "supervisor": asdict(supervisor_layout)
                }
            }, f, indent=2)
        
        # Save React components
        react_components = self.generate_react_components()
        react_dir = ui_dir / "react_components"
        react_dir.mkdir(exist_ok=True)
        
        for component_name, component_code in react_components.items():
            component_file = react_dir / f"{component_name}.jsx"
            with open(component_file, 'w') as f:
                f.write(component_code)
        
        logger.info(f"✅ UI Framework saved to {ui_dir}")
        return ui_dir

def main():
    """Generate complete Pixelated Empathy UI/UX framework"""
    logger.info("🎨 Generating Pixelated Empathy UI/UX Framework")
    
    ui_framework = PixelatedEmpathyUIFramework()
    
    # Create layouts
    trainee_layout = ui_framework.create_trainee_layout()
    supervisor_layout = ui_framework.create_supervisor_layout()
    
    # Generate React components
    react_components = ui_framework.generate_react_components()
    
    # Save framework
    saved_path = ui_framework.save_ui_framework()
    
    logger.info("🎯 UI/UX Framework Complete!")
    logger.info(f"📁 Framework saved to: {saved_path}")
    logger.info(f"🎭 Components created: {len(ui_framework.ui_components)}")
    logger.info(f"📱 React components generated: {len(react_components)}")

if __name__ == "__main__":
    main()