#!/usr/bin/env python3
"""
Conversation Interface Components
Specialized UI components for the therapeutic conversation experience.

Creates immersive, realistic conversation interface with AI client visualization,
real-time emotional feedback, and therapeutic guidance.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationInterfaceBuilder:
    """Builder for immersive conversation interface components"""
    
    def __init__(self):
        self.conversation_components = {}
        self._initialize_conversation_components()
    
    def _initialize_conversation_components(self):
        """Initialize all conversation-specific UI components"""
        
        # Main Chat Interface
        self.conversation_components["chat_interface"] = {
            "component_type": "ChatInterface",
            "react_code": self._generate_chat_interface_react(),
            "styling": self._generate_chat_interface_styles(),
            "features": [
                "Real-time message display",
                "Typing indicators",
                "Message composition",
                "Therapeutic response suggestions",
                "Crisis detection alerts",
                "Session timing"
            ]
        }
        
        # Client Avatar with Emotional Display
        self.conversation_components["emotional_avatar"] = {
            "component_type": "EmotionalAvatar", 
            "react_code": self._generate_emotional_avatar_react(),
            "styling": self._generate_avatar_styles(),
            "features": [
                "Dynamic facial expressions",
                "Body language indicators",
                "Voice tone visualization",
                "Eye contact patterns",
                "Stress level indicators",
                "Breakthrough moment celebrations"
            ]
        }
        
        # Message Composer with AI Assistance
        self.conversation_components["message_composer"] = {
            "component_type": "MessageComposer",
            "react_code": self._generate_message_composer_react(),
            "styling": self._generate_composer_styles(),
            "features": [
                "Real-time typing assistance",
                "Therapeutic response suggestions",
                "Intervention type selection",
                "Character count and guidance",
                "Send confidence indicator",
                "Emergency intervention button"
            ]
        }
        
        # Real-time Feedback Overlay
        self.conversation_components["feedback_overlay"] = {
            "component_type": "FeedbackOverlay",
            "react_code": self._generate_feedback_overlay_react(),
            "styling": self._generate_overlay_styles(),
            "features": [
                "Skill meter updates",
                "Therapeutic progress tracking",
                "Resistance level monitoring",
                "Crisis risk alerts",
                "Breakthrough opportunity highlights",
                "Supervisor notifications"
            ]
        }
    
    def _generate_chat_interface_react(self) -> str:
        """Generate React component for main chat interface"""
        return '''
import React, { useState, useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';
import { TypingIndicator } from './TypingIndicator';
import { EmergencyAlert } from './EmergencyAlert';

export const ChatInterface = ({ 
  conversationHistory, 
  onSendMessage, 
  clientState, 
  isClientTyping,
  emergencyAlerts 
}) => {
  const messagesEndRef = useRef(null);
  const [inputValue, setInputValue] = useState('');
  const [isComposing, setIsComposing] = useState(false);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  
  useEffect(scrollToBottom, [conversationHistory]);
  
  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;
    
    setIsComposing(true);
    await onSendMessage(inputValue);
    setInputValue('');
    setIsComposing(false);
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  return (
    <div className="chat-interface">
      {/* Emergency Alerts */}
      {emergencyAlerts?.map(alert => (
        <EmergencyAlert key={alert.id} alert={alert} />
      ))}
      
      {/* Session Header */}
      <div className="session-header">
        <div className="client-info">
          <span className="client-name">{clientState?.clientName}</span>
          <span className="session-time">{clientState?.sessionDuration}</span>
        </div>
        <div className="session-status">
          <span className={`status-indicator ${clientState?.emotionalState}`}>
            {clientState?.emotionalState}
          </span>
        </div>
      </div>
      
      {/* Message History */}
      <div className="message-history">
        {conversationHistory.map((message, index) => (
          <MessageBubble 
            key={index}
            message={message}
            clientState={clientState}
            showTimestamp={true}
            showEmotionalContext={message.type === 'client'}
          />
        ))}
        
        {isClientTyping && <TypingIndicator clientName={clientState?.clientName} />}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Message Input */}
      <div className="message-input-container">
        <MessageComposer
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSendMessage}
          onKeyPress={handleKeyPress}
          isComposing={isComposing}
          clientState={clientState}
          placeholder="Type your therapeutic response..."
        />
      </div>
    </div>
  );
};
'''
    
    def _generate_emotional_avatar_react(self) -> str:
        """Generate React component for emotional avatar"""
        return '''
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export const EmotionalAvatar = ({ 
  emotionalState, 
  nonverbalCues, 
  resistanceLevel,
  trustLevel,
  crisisRisk 
}) => {
  const [currentExpression, setCurrentExpression] = useState('neutral');
  const [bodyLanguage, setBodyLanguage] = useState('open');
  const [eyeContact, setEyeContact] = useState('direct');
  
  useEffect(() => {
    updateAvatarState(emotionalState, nonverbalCues, resistanceLevel);
  }, [emotionalState, nonverbalCues, resistanceLevel]);
  
  const updateAvatarState = (emotion, cues, resistance) => {
    // Determine facial expression
    if (resistance > 0.7) {
      setCurrentExpression('defensive');
      setBodyLanguage('closed');
      setEyeContact('avoidant');
    } else if (emotion === 'angry') {
      setCurrentExpression('hostile');
      setBodyLanguage('aggressive');
      setEyeContact('intense');
    } else if (emotion === 'sad') {
      setCurrentExpression('depressed');
      setBodyLanguage('withdrawn');
      setEyeContact('downcast');
    } else if (trustLevel > 0.6) {
      setCurrentExpression('engaged');
      setBodyLanguage('open');
      setEyeContact('direct');
    }
  };
  
  const getAvatarAnimation = () => {
    return {
      scale: trustLevel > 0.8 ? 1.05 : resistanceLevel > 0.8 ? 0.95 : 1.0,
      rotate: bodyLanguage === 'closed' ? -2 : bodyLanguage === 'open' ? 2 : 0,
      transition: { duration: 2, ease: "easeInOut" }
    };
  };
  
  const getEmotionalColor = () => {
    if (crisisRisk > 0.6) return '#ef4444'; // Crisis red
    if (resistanceLevel > 0.7) return '#f59e0b'; // Resistance amber
    if (trustLevel > 0.6) return '#10b981'; // Trust green
    return '#6b7280'; // Neutral gray
  };
  
  return (
    <div className="emotional-avatar-container">
      {/* Avatar Display */}
      <motion.div 
        className="avatar-display"
        animate={getAvatarAnimation()}
        style={{ 
          borderColor: getEmotionalColor(),
          backgroundColor: `${getEmotionalColor()}10`
        }}
      >
        {/* Face Expression */}
        <div className={`avatar-face expression-${currentExpression}`}>
          <div className={`eyes ${eyeContact}`}></div>
          <div className={`mouth expression-${currentExpression}`}></div>
        </div>
        
        {/* Body Language Indicator */}
        <div className={`body-posture ${bodyLanguage}`}></div>
      </motion.div>
      
      {/* Emotional State Indicators */}
      <div className="emotional-indicators">
        <div className="indicator-row">
          <span className="indicator-label">Emotional State:</span>
          <span className={`emotional-badge ${emotionalState}`}>
            {emotionalState}
          </span>
        </div>
        
        <div className="indicator-row">
          <span className="indicator-label">Trust Level:</span>
          <div className="trust-meter">
            <div 
              className="trust-fill" 
              style={{ width: `${trustLevel * 100}%` }}
            ></div>
          </div>
        </div>
        
        <div className="indicator-row">
          <span className="indicator-label">Resistance:</span>
          <div className="resistance-meter">
            <div 
              className="resistance-fill" 
              style={{ width: `${resistanceLevel * 100}%` }}
            ></div>
          </div>
        </div>
      </div>
      
      {/* Nonverbal Cues Display */}
      <div className="nonverbal-cues">
        <div className="cue-item">
          <span className="cue-label">Body Language:</span>
          <span className="cue-value">{nonverbalCues?.body_language}</span>
        </div>
        <div className="cue-item">
          <span className="cue-label">Voice Tone:</span>
          <span className="cue-value">{nonverbalCues?.voice_tone}</span>
        </div>
        <div className="cue-item">
          <span className="cue-label">Eye Contact:</span>
          <span className="cue-value">{nonverbalCues?.eye_contact}</span>
        </div>
      </div>
      
      {/* Crisis Risk Alert */}
      {crisisRisk > 0.6 && (
        <motion.div 
          className="crisis-alert"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          ⚠️ Monitor for Crisis Risk
        </motion.div>
      )}
    </div>
  );
};
'''
    
    def _generate_message_composer_react(self) -> str:
        """Generate React component for message composer"""
        return '''
import React, { useState, useEffect } from 'react';
import { SuggestionPanel } from './SuggestionPanel';
import { InterventionSelector } from './InterventionSelector';

export const MessageComposer = ({ 
  value, 
  onChange, 
  onSend, 
  onKeyPress,
  isComposing,
  clientState,
  placeholder 
}) => {
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [interventionType, setInterventionType] = useState('reflection');
  const [characterCount, setCharacterCount] = useState(0);
  const [therapeuticQuality, setTherapeuticQuality] = useState(0);
  
  useEffect(() => {
    setCharacterCount(value.length);
    assessTherapeuticQuality(value);
  }, [value]);
  
  const assessTherapeuticQuality = (message) => {
    // Simple quality assessment based on therapeutic indicators
    let quality = 0.5;
    
    const therapeuticWords = [
      'understand', 'feel', 'sounds like', 'tell me more',
      'what was that like', 'how did you', 'i hear you'
    ];
    
    const problematicWords = [
      'should', 'must', 'have to', 'wrong', 'bad'
    ];
    
    therapeuticWords.forEach(word => {
      if (message.toLowerCase().includes(word)) quality += 0.1;
    });
    
    problematicWords.forEach(word => {
      if (message.toLowerCase().includes(word)) quality -= 0.1;
    });
    
    setTherapeuticQuality(Math.max(0, Math.min(1, quality)));
  };
  
  const getSuggestions = () => {
    if (!clientState) return [];
    
    const suggestions = [];
    
    if (clientState.resistanceLevel > 0.7) {
      suggestions.push({
        type: 'reflection',
        text: "I can see this feels difficult to talk about.",
        rationale: "Acknowledge resistance without pushing"
      });
      suggestions.push({
        type: 'validation',
        text: "You have every right to feel cautious about sharing.",
        rationale: "Validate client's autonomy"
      });
    }
    
    if (clientState.emotionalState === 'angry') {
      suggestions.push({
        type: 'empathy',
        text: "I can hear how frustrated you are right now.",
        rationale: "Validate the emotion"
      });
      suggestions.push({
        type: 'inquiry',
        text: "Help me understand what's making you feel this way.",
        rationale: "Explore the anger source"
      });
    }
    
    if (clientState.crisisRisk > 0.6) {
      suggestions.push({
        type: 'safety',
        text: "I'm concerned about your safety. Are you having thoughts of hurting yourself?",
        rationale: "Direct safety assessment needed"
      });
    }
    
    return suggestions;
  };
  
  const getQualityColor = () => {
    if (therapeuticQuality >= 0.8) return '#10b981'; // Excellent
    if (therapeuticQuality >= 0.6) return '#f59e0b'; // Good
    if (therapeuticQuality >= 0.4) return '#ef4444'; // Needs improvement
    return '#6b7280'; // Neutral
  };
  
  return (
    <div className="message-composer">
      {/* Intervention Type Selector */}
      <InterventionSelector 
        selectedType={interventionType}
        onTypeChange={setInterventionType}
        clientState={clientState}
      />
      
      {/* Main Text Input */}
      <div className="composer-main">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyPress={onKeyPress}
          placeholder={placeholder}
          className="message-input"
          rows={3}
          disabled={isComposing}
        />
        
        {/* Input Assistance */}
        <div className="input-assistance">
          <div className="character-count">
            <span style={{ color: characterCount > 500 ? '#ef4444' : '#6b7280' }}>
              {characterCount}/1000
            </span>
          </div>
          
          <div className="therapeutic-quality">
            <span>Quality: </span>
            <div 
              className="quality-indicator"
              style={{ 
                backgroundColor: getQualityColor(),
                width: `${therapeuticQuality * 100}%`
              }}
            ></div>
          </div>
        </div>
      </div>
      
      {/* Action Buttons */}
      <div className="composer-actions">
        <button 
          className="suggestion-toggle"
          onClick={() => setShowSuggestions(!showSuggestions)}
          disabled={isComposing}
        >
          💡 Suggestions
        </button>
        
        <button 
          className="emergency-button"
          onClick={() => onSend("EMERGENCY_INTERVENTION_NEEDED")}
        >
          🚨 Emergency
        </button>
        
        <button 
          className="send-button"
          onClick={onSend}
          disabled={!value.trim() || isComposing}
        >
          {isComposing ? '⏳ Sending...' : '📤 Send'}
        </button>
      </div>
      
      {/* Therapeutic Suggestions Panel */}
      {showSuggestions && (
        <SuggestionPanel 
          suggestions={getSuggestions()}
          onSelectSuggestion={(suggestion) => {
            onChange(suggestion.text);
            setShowSuggestions(false);
          }}
          clientState={clientState}
        />
      )}
    </div>
  );
};
'''
    
    def _generate_chat_interface_styles(self) -> str:
        """Generate CSS styles for chat interface"""
        return '''
.chat-interface {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #ffffff;
  border: 2px solid #3b82f6;
  border-radius: 12px;
  overflow: hidden;
}

.session-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background: linear-gradient(135deg, #3b82f6, #1d4ed8);
  color: white;
  border-bottom: 1px solid #e5e7eb;
}

.client-info {
  display: flex;
  flex-direction: column;
}

.client-name {
  font-weight: 600;
  font-size: 1.1rem;
}

.session-time {
  font-size: 0.9rem;
  opacity: 0.9;
}

.session-status {
  display: flex;
  align-items: center;
}

.status-indicator {
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
  text-transform: capitalize;
}

.status-indicator.calm { background: #dcfce7; color: #166534; }
.status-indicator.anxious { background: #fef3c7; color: #92400e; }
.status-indicator.angry { background: #fee2e2; color: #991b1b; }
.status-indicator.sad { background: #dbeafe; color: #1e40af; }
.status-indicator.resistant { background: #f3e8ff; color: #7c2d12; }

.message-history {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background: #f9fafb;
}

.message-input-container {
  border-top: 1px solid #e5e7eb;
  background: white;
}

/* Emergency Alert Styles */
.emergency-alert {
  background: linear-gradient(135deg, #ef4444, #dc2626);
  color: white;
  padding: 1rem;
  border-left: 4px solid #b91c1c;
  margin-bottom: 1rem;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.8; }
}

/* Responsive Design */
@media (max-width: 768px) {
  .session-header {
    padding: 0.75rem;
  }
  
  .client-name {
    font-size: 1rem;
  }
  
  .message-history {
    padding: 0.75rem;
  }
}
'''
    
    def save_conversation_components(self) -> Dict[str, str]:
        """Save all conversation interface components"""
        
        ui_dir = Path("ai/platform/ui")
        ui_dir.mkdir(exist_ok=True)
        
        components_dir = ui_dir / "conversation_components"
        components_dir.mkdir(exist_ok=True)
        
        # Save React components
        for component_name, component_data in self.conversation_components.items():
            
            # Save React component
            react_file = components_dir / f"{component_data['component_type']}.jsx"
            with open(react_file, 'w') as f:
                f.write(component_data['react_code'])
            
            # Save styles
            if 'styling' in component_data:
                css_file = components_dir / f"{component_data['component_type']}.css"
                with open(css_file, 'w') as f:
                    f.write(component_data['styling'])
        
        # Save component documentation
        docs_file = components_dir / "README.md"
        with open(docs_file, 'w') as f:
            f.write(self._generate_component_documentation())
        
        logger.info(f"✅ Conversation components saved to {components_dir}")
        
        return {
            "components_saved": len(self.conversation_components),
            "output_directory": str(components_dir),
            "react_components": list(self.conversation_components.keys())
        }
    
    def _generate_component_documentation(self) -> str:
        """Generate documentation for conversation components"""
        return '''# Pixelated Empathy Conversation Interface Components

## Overview
These components create an immersive, realistic therapeutic conversation experience between trainees and AI clients.

## Components

### ChatInterface
- **Purpose**: Main conversation interface with message history and real-time updates
- **Features**: 
  - Real-time message display
  - Typing indicators
  - Emergency alerts
  - Session timing
  - Emotional state tracking

### EmotionalAvatar
- **Purpose**: Visual representation of client's emotional state and nonverbal cues
- **Features**:
  - Dynamic facial expressions
  - Body language indicators
  - Trust/resistance meters
  - Crisis risk alerts
  - Smooth animations

### MessageComposer
- **Purpose**: Intelligent message composition with therapeutic guidance
- **Features**:
  - Real-time quality assessment
  - Therapeutic response suggestions
  - Intervention type selection
  - Character count and guidance
  - Emergency intervention button

### FeedbackOverlay
- **Purpose**: Real-time skill feedback and progress tracking
- **Features**:
  - Skill meter updates
  - Progress indicators
  - Crisis alerts
  - Breakthrough celebrations
  - Supervisor notifications

## Usage
Import components into main Pixelated Empathy platform:

```jsx
import { ChatInterface } from './ChatInterface';
import { EmotionalAvatar } from './EmotionalAvatar';
import { MessageComposer } from './MessageComposer';
```

## Styling
Each component includes responsive CSS with the Pixelated Empathy theme colors and professional therapeutic design.
'''

def main():
    """Generate conversation interface components"""
    logger.info("🎨 Generating Conversation Interface Components")
    
    builder = ConversationInterfaceBuilder()
    result = builder.save_conversation_components()
    
    logger.info("✅ Conversation Interface Components Complete!")
    logger.info(f"📱 Components created: {result['components_saved']}")
    logger.info(f"📁 Saved to: {result['output_directory']}")

if __name__ == "__main__":
    main()