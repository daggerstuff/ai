#!/usr/bin/env python3
"""
Supervisor Dashboard Components
Real-time evaluation and monitoring interface for supervisors.

Creates comprehensive dashboard for supervising therapeutic training sessions
with live assessment tools, intervention capabilities, and analytics.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupervisorDashboardBuilder:
    """Builder for supervisor dashboard UI components"""
    
    def __init__(self):
        self.dashboard_components = {}
        self._initialize_supervisor_components()
    
    def _initialize_supervisor_components(self):
        """Initialize all supervisor dashboard components"""
        
        # Real-time Session Monitor
        self.dashboard_components["session_monitor"] = {
            "component_type": "SessionMonitor",
            "react_code": self._generate_session_monitor_react(),
            "styling": self._generate_monitor_styles(),
            "features": [
                "Live conversation viewing",
                "Real-time emotional state tracking",
                "Skill performance monitoring",
                "Crisis alert notifications",
                "Session recording controls",
                "Multi-session support"
            ]
        }
        
        # Competency Assessment Grid
        self.dashboard_components["competency_grid"] = {
            "component_type": "CompetencyGrid",
            "react_code": self._generate_competency_grid_react(),
            "styling": self._generate_grid_styles(),
            "features": [
                "8 core therapeutic skills",
                "5-level rating system",
                "Real-time skill updates",
                "Evidence collection",
                "Rubric integration",
                "Progress tracking"
            ]
        }
    
    def _generate_session_monitor_react(self) -> str:
        """Generate React component for session monitoring"""
        return '''
import React, { useState, useEffect } from 'react';
import { CompetencyRating } from './CompetencyRating';
import { InterventionPanel } from './InterventionPanel';
import { CrisisAlert } from './CrisisAlert';

export const SessionMonitor = ({ 
  sessionId, 
  conversationHistory, 
  clientState, 
  traineePerformance,
  onRecordObservation,
  onTriggerIntervention 
}) => {
  const [observations, setObservations] = useState([]);
  const [currentSkillFocus, setCurrentSkillFocus] = useState('rapport_building');
  const [interventionQueue, setInterventionQueue] = useState([]);
  
  useEffect(() => {
    // Monitor for automatic alerts
    checkForInterventionNeeds(clientState, traineePerformance);
  }, [clientState, traineePerformance]);
  
  const checkForInterventionNeeds = (clientState, performance) => {
    const alerts = [];
    
    if (clientState?.crisisRisk > 0.6) {
      alerts.push({
        type: 'crisis',
        priority: 'immediate',
        message: 'Crisis risk detected - immediate intervention needed'
      });
    }
    
    if (performance?.overallCompetency < 2.0) {
      alerts.push({
        type: 'skill_deficit',
        priority: 'high', 
        message: 'Significant skill deficits - consider session break'
      });
    }
    
    setInterventionQueue(alerts);
  };
  
  const recordObservation = (skillArea, rating, evidence) => {
    const observation = {
      timestamp: new Date(),
      skillArea,
      rating,
      evidence,
      sessionPhase: clientState?.sessionPhase
    };
    
    setObservations(prev => [...prev, observation]);
    onRecordObservation(observation);
  };
  
  return (
    <div className="session-monitor">
      {/* Crisis Alerts */}
      {interventionQueue.map(alert => (
        <CrisisAlert 
          key={alert.type}
          alert={alert}
          onIntervene={() => onTriggerIntervention(alert)}
        />
      ))}
      
      {/* Main Monitor Layout */}
      <div className="monitor-layout">
        {/* Left: Conversation View */}
        <div className="conversation-monitor">
          <div className="monitor-header">
            <h3>Live Session: {sessionId}</h3>
            <div className="session-indicators">
              <span className={`status-indicator ${clientState?.emotionalState}`}>
                {clientState?.emotionalState}
              </span>
              <span className="duration">
                {clientState?.sessionDuration}
              </span>
            </div>
          </div>
          
          <div className="conversation-history">
            {conversationHistory.map((message, index) => (
              <div key={index} className={`message-item ${message.type}`}>
                <div className="message-header">
                  <span className="speaker">
                    {message.type === 'therapist' ? 'Trainee' : 'Client'}
                  </span>
                  <span className="timestamp">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="message-content">
                  {message.content}
                </div>
                {message.type === 'therapist' && (
                  <div className="assessment-overlay">
                    <CompetencyRating
                      messageId={index}
                      onRate={(skill, rating) => recordObservation(skill, rating, message.content)}
                      currentSkillFocus={currentSkillFocus}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
        
        {/* Right: Assessment Panel */}
        <div className="assessment-panel">
          <div className="skill-selector">
            <h4>Focus Skill Assessment</h4>
            <select 
              value={currentSkillFocus}
              onChange={(e) => setCurrentSkillFocus(e.target.value)}
            >
              <option value="rapport_building">Rapport Building</option>
              <option value="active_listening">Active Listening</option>
              <option value="empathy">Empathy Demonstration</option>
              <option value="crisis_management">Crisis Management</option>
              <option value="resistance_handling">Resistance Handling</option>
              <option value="boundary_setting">Boundary Setting</option>
            </select>
          </div>
          
          <div className="current-performance">
            <h4>Current Performance</h4>
            <div className="performance-metrics">
              <div className="metric">
                <span>Trust Building:</span>
                <div className="progress-bar">
                  <div 
                    className="progress-fill trust"
                    style={{ width: `${(clientState?.trustLevel || 0) * 100}%` }}
                  ></div>
                </div>
              </div>
              <div className="metric">
                <span>Resistance Level:</span>
                <div className="progress-bar">
                  <div 
                    className="progress-fill resistance"
                    style={{ width: `${(clientState?.resistanceLevel || 0) * 100}%` }}
                  ></div>
                </div>
              </div>
              <div className="metric">
                <span>Overall Competency:</span>
                <div className="competency-score">
                  {traineePerformance?.overallCompetency?.toFixed(1) || 'N/A'}/5.0
                </div>
              </div>
            </div>
          </div>
          
          <div className="recent-observations">
            <h4>Recent Observations</h4>
            <div className="observations-list">
              {observations.slice(-5).map((obs, index) => (
                <div key={index} className="observation-item">
                  <div className="obs-header">
                    <span className="skill">{obs.skillArea}</span>
                    <span className="rating">Rating: {obs.rating}/5</span>
                  </div>
                  <div className="obs-evidence">
                    {obs.evidence.substring(0, 100)}...
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};'''
    
    def _generate_competency_grid_react(self) -> str:
        """Generate React component for competency assessment grid"""
        return '''
import React, { useState, useEffect } from 'react';
import { SkillRubric } from './SkillRubric';
import { EvidenceCollector } from './EvidenceCollector';

export const CompetencyGrid = ({ 
  traineeId,
  sessionData,
  skillRatings,
  onUpdateRating,
  onViewRubric,
  onCollectEvidence
}) => {
  const [selectedSkill, setSelectedSkill] = useState(null);
  const [showRubric, setShowRubric] = useState(false);
  const [evidenceMode, setEvidenceMode] = useState(false);
  
  const skillCategories = [
    {
      category: "Core Therapeutic Skills",
      skills: [
        { id: "rapport_building", name: "Rapport Building", current: skillRatings?.rapport_building || 0 },
        { id: "active_listening", name: "Active Listening", current: skillRatings?.active_listening || 0 },
        { id: "empathy_demonstration", name: "Empathy Demonstration", current: skillRatings?.empathy || 0 },
        { id: "therapeutic_response", name: "Therapeutic Response", current: skillRatings?.therapeutic_response || 0 }
      ]
    },
    {
      category: "Advanced Skills",
      skills: [
        { id: "crisis_management", name: "Crisis Management", current: skillRatings?.crisis_management || 0 },
        { id: "resistance_handling", name: "Resistance Handling", current: skillRatings?.resistance_handling || 0 },
        { id: "boundary_setting", name: "Boundary Setting", current: skillRatings?.boundary_setting || 0 },
        { id: "cultural_competence", name: "Cultural Competence", current: skillRatings?.cultural_competence || 0 }
      ]
    }
  ];
  
  const getCompetencyLabel = (rating) => {
    if (rating >= 4.5) return "Exemplary";
    if (rating >= 3.5) return "Proficient"; 
    if (rating >= 2.5) return "Competent";
    if (rating >= 1.5) return "Developing";
    return "Unsatisfactory";
  };
  
  const getCompetencyColor = (rating) => {
    if (rating >= 4.5) return "#10b981"; // Exemplary - Green
    if (rating >= 3.5) return "#3b82f6"; // Proficient - Blue
    if (rating >= 2.5) return "#f59e0b"; // Competent - Amber
    if (rating >= 1.5) return "#ef4444"; // Developing - Red
    return "#6b7280"; // Unsatisfactory - Gray
  };
  
  const handleRatingChange = (skillId, newRating) => {
    onUpdateRating(skillId, newRating);
  };
  
  const handleSkillClick = (skill) => {
    setSelectedSkill(skill);
    setShowRubric(true);
  };
  
  return (
    <div className="competency-grid">
      <div className="grid-header">
        <h3>Skill Competency Assessment</h3>
        <div className="grid-controls">
          <button 
            className={`mode-button ${evidenceMode ? 'active' : ''}`}
            onClick={() => setEvidenceMode(!evidenceMode)}
          >
            📝 Evidence Mode
          </button>
          <button 
            className="export-button"
            onClick={() => onExportAssessment()}
          >
            📊 Export
          </button>
        </div>
      </div>
      
      <div className="competency-categories">
        {skillCategories.map(category => (
          <div key={category.category} className="skill-category">
            <h4 className="category-title">{category.category}</h4>
            
            <div className="skills-grid">
              {category.skills.map(skill => (
                <div key={skill.id} className="skill-item">
                  <div className="skill-header">
                    <span 
                      className="skill-name"
                      onClick={() => handleSkillClick(skill)}
                    >
                      {skill.name}
                    </span>
                    <span 
                      className="competency-badge"
                      style={{ backgroundColor: getCompetencyColor(skill.current) }}
                    >
                      {getCompetencyLabel(skill.current)}
                    </span>
                  </div>
                  
                  <div className="rating-controls">
                    <div className="rating-scale">
                      {[1, 2, 3, 4, 5].map(rating => (
                        <button
                          key={rating}
                          className={`rating-button ${skill.current === rating ? 'active' : ''}`}
                          onClick={() => handleRatingChange(skill.id, rating)}
                          disabled={!evidenceMode}
                        >
                          {rating}
                        </button>
                      ))}
                    </div>
                    
                    <div className="rating-visual">
                      <div className="rating-bar">
                        <div 
                          className="rating-fill"
                          style={{ 
                            width: `${(skill.current / 5) * 100}%`,
                            backgroundColor: getCompetencyColor(skill.current)
                          }}
                        ></div>
                      </div>
                      <span className="rating-value">{skill.current.toFixed(1)}/5.0</span>
                    </div>
                  </div>
                  
                  {evidenceMode && (
                    <div className="evidence-section">
                      <button 
                        className="collect-evidence-btn"
                        onClick={() => onCollectEvidence(skill.id)}
                      >
                        + Add Evidence
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      
      {/* Overall Assessment Summary */}
      <div className="assessment-summary">
        <h4>Session Summary</h4>
        <div className="summary-metrics">
          <div className="metric-item">
            <span className="metric-label">Overall Competency:</span>
            <span className="metric-value">
              {(Object.values(skillRatings || {}).reduce((a, b) => a + b, 0) / 
                Object.keys(skillRatings || {}).length || 0).toFixed(1)}/5.0
            </span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Skills Above Target:</span>
            <span className="metric-value">
              {Object.values(skillRatings || {}).filter(rating => rating >= 3.0).length}/
              {Object.keys(skillRatings || {}).length}
            </span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Readiness for Independent Practice:</span>
            <span className={`readiness-indicator ${
              Object.values(skillRatings || {}).every(rating => rating >= 3.0) ? 'ready' : 'not-ready'
            }`}>
              {Object.values(skillRatings || {}).every(rating => rating >= 3.0) ? 'Ready' : 'Needs Development'}
            </span>
          </div>
        </div>
      </div>
      
      {/* Skill Rubric Modal */}
      {showRubric && selectedSkill && (
        <SkillRubric 
          skill={selectedSkill}
          onClose={() => setShowRubric(false)}
          onRate={(rating) => {
            handleRatingChange(selectedSkill.id, rating);
            setShowRubric(false);
          }}
        />
      )}
    </div>
  );
};'''