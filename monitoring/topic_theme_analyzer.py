#!/usr/bin/env python3
"""
Conversation Topic and Theme Analysis System
Analyzes conversation topics, themes, and provides insights about content patterns
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TopicAnalysis:
    """Topic analysis results"""
    topic_name: str
    frequency: int
    percentage: float
    keywords: List[str]
    related_themes: List[str]
    conversation_examples: List[str]
    quality_score: float
    datasets: List[str]

@dataclass
class ThemeAnalysis:
    """Theme analysis results"""
    theme_name: str
    topic_clusters: List[str]
    conversation_count: int
    emotional_indicators: Dict[str, int]
    complexity_score: float
    therapeutic_relevance: float
    common_patterns: List[str]

class TopicThemeAnalyzer:
    """Enterprise-grade topic and theme analysis system"""
    
    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/topic_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis configuration
        self.analysis_config = {
            'max_conversations_per_dataset': 500,
            'min_topic_frequency': 5,
            'topic_keywords': {
                'mental_health': ['anxiety', 'depression', 'stress', 'mental', 'emotional', 'therapy', 'counseling'],
                'relationships': ['relationship', 'partner', 'family', 'friend', 'love', 'marriage', 'dating'],
                'work_career': ['work', 'job', 'career', 'workplace', 'professional', 'employment', 'boss'],
                'health_wellness': ['health', 'wellness', 'exercise', 'diet', 'sleep', 'medical', 'doctor'],
                'personal_growth': ['growth', 'development', 'learning', 'goals', 'motivation', 'success'],
                'life_challenges': ['problem', 'challenge', 'difficulty', 'struggle', 'issue', 'conflict'],
                'emotions': ['feel', 'emotion', 'happy', 'sad', 'angry', 'frustrated', 'excited', 'worried'],
                'communication': ['talk', 'speak', 'listen', 'communicate', 'conversation', 'discuss'],
                'decision_making': ['decision', 'choice', 'option', 'decide', 'consider', 'think'],
                'support_help': ['help', 'support', 'advice', 'guidance', 'assistance', 'recommend']
            },
            'therapeutic_themes': {
                'cognitive_behavioral': ['thought', 'thinking', 'belief', 'behavior', 'pattern', 'habit'],
                'mindfulness': ['mindful', 'present', 'awareness', 'meditation', 'breathing', 'focus'],
                'trauma_recovery': ['trauma', 'recovery', 'healing', 'past', 'memory', 'overcome'],
                'coping_strategies': ['cope', 'manage', 'handle', 'deal', 'strategy', 'technique'],
                'self_esteem': ['confidence', 'self-worth', 'value', 'worth', 'self-esteem', 'identity']
            }
        }
        
    def analyze_topics_and_themes(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze conversation topics and themes"""
        print("🔍 Analyzing conversation topics and themes...")
        
        try:
            # Get conversation data
            conversations = self._get_conversation_data(dataset_name)
            
            if not conversations:
                print("❌ No conversation data found")
                return {}
            
            # Extract topics
            topic_analyses = self._analyze_topics(conversations)
            
            # Extract themes
            theme_analyses = self._analyze_themes(conversations)
            
            # Generate insights
            insights = self._generate_topic_theme_insights(topic_analyses, theme_analyses)
            
            results = {
                'topics': topic_analyses,
                'themes': theme_analyses,
                'insights': insights,
                'total_conversations_analyzed': len(conversations)
            }
            
            print(f"✅ Analyzed {len(topic_analyses)} topics and {len(theme_analyses)} themes from {len(conversations)} conversations")
            return results
            
        except Exception as e:
            print(f"❌ Error analyzing topics and themes: {e}")
            return {}
    
    def _get_conversation_data(self, dataset_name: Optional[str] = None) -> List[Dict]:
        """Get conversation data for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if dataset_name:
                query = """
                SELECT 
                    conversation_id,
                    dataset_source,
                    conversations_json,
                    word_count,
                    tier
                FROM conversations 
                WHERE dataset_source = ?
                AND conversations_json IS NOT NULL
                AND word_count >= 20
                ORDER BY RANDOM()
                LIMIT ?
                """
                params = (dataset_name, self.analysis_config['max_conversations_per_dataset'])
            else:
                query = """
                SELECT 
                    conversation_id,
                    dataset_source,
                    conversations_json,
                    word_count,
                    tier
                FROM conversations 
                WHERE conversations_json IS NOT NULL
                AND word_count >= 20
                ORDER BY RANDOM()
                LIMIT ?
                """
                params = (self.analysis_config['max_conversations_per_dataset'] * 3,)
            
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            conversations = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                try:
                    record['conversations'] = json.loads(record['conversations_json'])
                    conversations.append(record)
                except json.JSONDecodeError:
                    continue
            
            conn.close()
            return conversations
            
        except Exception as e:
            print(f"❌ Error getting conversation data: {e}")
            return []
    
    def _analyze_topics(self, conversations: List[Dict]) -> Dict[str, TopicAnalysis]:
        """Analyze topics in conversations"""
        try:
            topic_counts = defaultdict(int)
            topic_conversations = defaultdict(list)
            topic_datasets = defaultdict(set)
            topic_keywords_found = defaultdict(set)
            
            # Analyze each conversation
            for conv_data in conversations:
                conversation_text = self._extract_conversation_text(conv_data['conversations'])
                if not conversation_text:
                    continue
                
                text_lower = conversation_text.lower()
                conv_id = conv_data['conversation_id']
                dataset = conv_data['dataset_source']
                
                # Check for each topic
                for topic_name, keywords in self.analysis_config['topic_keywords'].items():
                    found_keywords = []
                    for keyword in keywords:
                        if keyword in text_lower:
                            found_keywords.append(keyword)
                    
                    if found_keywords:
                        topic_counts[topic_name] += 1
                        topic_conversations[topic_name].append(conv_id)
                        topic_datasets[topic_name].add(dataset)
                        topic_keywords_found[topic_name].update(found_keywords)
            
            # Create topic analyses
            total_conversations = len(conversations)
            topic_analyses = {}
            
            for topic_name, count in topic_counts.items():
                if count >= self.analysis_config['min_topic_frequency']:
                    percentage = (count / total_conversations) * 100
                    
                    # Generate related themes
                    related_themes = self._find_related_themes(topic_name)
                    
                    # Calculate quality score
                    quality_score = min(1.0, count / 100)  # Normalize to max 100 conversations
                    
                    topic_analyses[topic_name] = TopicAnalysis(
                        topic_name=topic_name,
                        frequency=count,
                        percentage=percentage,
                        keywords=list(topic_keywords_found[topic_name]),
                        related_themes=related_themes,
                        conversation_examples=topic_conversations[topic_name][:5],  # First 5 examples
                        quality_score=quality_score,
                        datasets=list(topic_datasets[topic_name])
                    )
            
            return topic_analyses
            
        except Exception as e:
            print(f"❌ Error analyzing topics: {e}")
            return {}
    
    def _analyze_themes(self, conversations: List[Dict]) -> Dict[str, ThemeAnalysis]:
        """Analyze themes in conversations"""
        try:
            theme_counts = defaultdict(int)
            theme_topics = defaultdict(set)
            theme_emotions = defaultdict(lambda: defaultdict(int))
            
            # Analyze each conversation
            for conv_data in conversations:
                conversation_text = self._extract_conversation_text(conv_data['conversations'])
                if not conversation_text:
                    continue
                
                text_lower = conversation_text.lower()
                
                # Check for therapeutic themes
                for theme_name, keywords in self.analysis_config['therapeutic_themes'].items():
                    found_keywords = []
                    for keyword in keywords:
                        if keyword in text_lower:
                            found_keywords.append(keyword)
                    
                    if found_keywords:
                        theme_counts[theme_name] += 1
                        
                        # Find related topics
                        for topic_name, topic_keywords in self.analysis_config['topic_keywords'].items():
                            if any(kw in text_lower for kw in topic_keywords):
                                theme_topics[theme_name].add(topic_name)
                        
                        # Analyze emotional indicators
                        emotion_words = ['happy', 'sad', 'angry', 'anxious', 'excited', 'worried', 'calm', 'stressed']
                        for emotion in emotion_words:
                            if emotion in text_lower:
                                theme_emotions[theme_name][emotion] += 1
            
            # Create theme analyses
            theme_analyses = {}
            
            for theme_name, count in theme_counts.items():
                if count >= self.analysis_config['min_topic_frequency']:
                    # Calculate complexity score
                    complexity_score = min(1.0, len(theme_topics[theme_name]) / 5)  # Normalize to max 5 topics
                    
                    # Calculate therapeutic relevance
                    therapeutic_relevance = min(1.0, count / 50)  # Normalize to max 50 conversations
                    
                    # Generate common patterns
                    common_patterns = self._identify_theme_patterns(theme_name, count)
                    
                    theme_analyses[theme_name] = ThemeAnalysis(
                        theme_name=theme_name,
                        topic_clusters=list(theme_topics[theme_name]),
                        conversation_count=count,
                        emotional_indicators=dict(theme_emotions[theme_name]),
                        complexity_score=complexity_score,
                        therapeutic_relevance=therapeutic_relevance,
                        common_patterns=common_patterns
                    )
            
            return theme_analyses
            
        except Exception as e:
            print(f"❌ Error analyzing themes: {e}")
            return {}
    
    def _extract_conversation_text(self, conversations: List) -> str:
        """Extract text from conversation structure"""
        try:
            all_text = ""
            
            for turn in conversations:
                if isinstance(turn, dict):
                    # Handle different conversation formats
                    text = ""
                    text += turn.get('human', '') + ' '
                    text += turn.get('assistant', '') + ' '
                    text += turn.get('user', '') + ' '
                    text += turn.get('bot', '') + ' '
                    text += turn.get('input', '') + ' '
                    text += turn.get('output', '') + ' '
                    all_text += text
                elif isinstance(turn, str):
                    all_text += turn + ' '
            
            return all_text.strip()
            
        except Exception as e:
            print(f"❌ Error extracting conversation text: {e}")
            return ""
    
    def _find_related_themes(self, topic_name: str) -> List[str]:
        """Find themes related to a topic"""
        related_themes = []
        
        # Define topic-theme relationships
        topic_theme_map = {
            'mental_health': ['cognitive_behavioral', 'mindfulness', 'coping_strategies'],
            'relationships': ['communication', 'self_esteem'],
            'work_career': ['coping_strategies', 'self_esteem'],
            'emotions': ['mindfulness', 'coping_strategies'],
            'life_challenges': ['coping_strategies', 'trauma_recovery'],
            'support_help': ['cognitive_behavioral', 'coping_strategies']
        }
        
        return topic_theme_map.get(topic_name, [])
    
    def _identify_theme_patterns(self, theme_name: str, count: int) -> List[str]:
        """Identify common patterns for a theme"""
        patterns = []
        
        # Theme-specific patterns
        if theme_name == 'cognitive_behavioral':
            patterns = ['Thought pattern analysis', 'Behavior modification focus', 'Belief system exploration']
        elif theme_name == 'mindfulness':
            patterns = ['Present moment awareness', 'Breathing techniques', 'Meditation practices']
        elif theme_name == 'trauma_recovery':
            patterns = ['Past event processing', 'Healing journey focus', 'Recovery milestone tracking']
        elif theme_name == 'coping_strategies':
            patterns = ['Problem-solving approaches', 'Stress management techniques', 'Adaptive behavior development']
        elif theme_name == 'self_esteem':
            patterns = ['Self-worth exploration', 'Confidence building', 'Identity development']
        
        # Add frequency-based patterns
        if count > 50:
            patterns.append('High frequency theme - core therapeutic focus')
        elif count > 20:
            patterns.append('Moderate frequency theme - regular therapeutic element')
        else:
            patterns.append('Emerging theme - specialized therapeutic area')
        
        return patterns
    
    def _generate_topic_theme_insights(self, topics: Dict[str, TopicAnalysis], 
                                     themes: Dict[str, ThemeAnalysis]) -> List[str]:
        """Generate insights from topic and theme analysis"""
        insights = []
        
        try:
            # Topic insights
            if topics:
                most_common_topic = max(topics.items(), key=lambda x: x[1].frequency)
                insights.append(f"Most common topic: '{most_common_topic[0]}' appears in {most_common_topic[1].frequency} conversations ({most_common_topic[1].percentage:.1f}%)")
                
                # Topic diversity
                topic_count = len(topics)
                insights.append(f"Topic diversity: {topic_count} distinct topics identified")
                
                # High-frequency topics
                high_freq_topics = [name for name, analysis in topics.items() if analysis.frequency > 20]
                if high_freq_topics:
                    insights.append(f"High-frequency topics: {', '.join(high_freq_topics)}")
            
            # Theme insights
            if themes:
                most_relevant_theme = max(themes.items(), key=lambda x: x[1].therapeutic_relevance)
                insights.append(f"Most therapeutically relevant theme: '{most_relevant_theme[0]}' with {most_relevant_theme[1].conversation_count} conversations")
                
                # Theme complexity
                complex_themes = [name for name, analysis in themes.items() if analysis.complexity_score > 0.6]
                if complex_themes:
                    insights.append(f"Complex themes with multiple topic clusters: {', '.join(complex_themes)}")
            
            # Cross-analysis insights
            if topics and themes:
                insights.append(f"Analysis coverage: {len(topics)} topics and {len(themes)} therapeutic themes identified")
                
                # Find most connected themes
                theme_connections = {name: len(analysis.topic_clusters) for name, analysis in themes.items()}
                if theme_connections:
                    most_connected = max(theme_connections.items(), key=lambda x: x[1])
                    insights.append(f"Most connected theme: '{most_connected[0]}' relates to {most_connected[1]} different topics")
            
            if not insights:
                insights.append("Insufficient data for comprehensive topic and theme analysis")
            
            return insights
            
        except Exception as e:
            print(f"❌ Error generating insights: {e}")
            return ["Error generating topic and theme insights"]
    
    def export_topic_theme_report(self, analysis_results: Dict[str, Any]) -> str:
        """Export comprehensive topic and theme analysis report"""
        print("📄 Exporting topic and theme analysis report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"topic_theme_analysis_report_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'analyzer_version': '1.0.0',
                    'analysis_type': 'topic_and_theme_analysis',
                    'conversations_analyzed': analysis_results.get('total_conversations_analyzed', 0)
                },
                'topic_analysis': {
                    name: {
                        'frequency': analysis.frequency,
                        'percentage': analysis.percentage,
                        'keywords': analysis.keywords,
                        'related_themes': analysis.related_themes,
                        'conversation_examples': analysis.conversation_examples,
                        'quality_score': analysis.quality_score,
                        'datasets': analysis.datasets
                    }
                    for name, analysis in analysis_results.get('topics', {}).items()
                },
                'theme_analysis': {
                    name: {
                        'topic_clusters': analysis.topic_clusters,
                        'conversation_count': analysis.conversation_count,
                        'emotional_indicators': analysis.emotional_indicators,
                        'complexity_score': analysis.complexity_score,
                        'therapeutic_relevance': analysis.therapeutic_relevance,
                        'common_patterns': analysis.common_patterns
                    }
                    for name, analysis in analysis_results.get('themes', {}).items()
                },
                'insights': analysis_results.get('insights', []),
                'summary_statistics': self._create_summary_statistics(analysis_results)
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported topic and theme analysis report to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
            return ""
    
    def _create_summary_statistics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics"""
        try:
            topics = analysis_results.get('topics', {})
            themes = analysis_results.get('themes', {})
            
            # Topic statistics
            topic_frequencies = [analysis.frequency for analysis in topics.values()]
            avg_topic_frequency = np.mean(topic_frequencies) if topic_frequencies else 0
            
            # Theme statistics
            theme_counts = [analysis.conversation_count for analysis in themes.values()]
            avg_theme_count = np.mean(theme_counts) if theme_counts else 0
            
            # Coverage statistics
            total_conversations = analysis_results.get('total_conversations_analyzed', 0)
            topic_coverage = sum(topic_frequencies) / total_conversations if total_conversations > 0 else 0
            
            return {
                'total_topics_identified': len(topics),
                'total_themes_identified': len(themes),
                'average_topic_frequency': float(avg_topic_frequency),
                'average_theme_count': float(avg_theme_count),
                'topic_coverage_rate': float(topic_coverage),
                'most_frequent_topic': max(topics.items(), key=lambda x: x[1].frequency)[0] if topics else None,
                'most_relevant_theme': max(themes.items(), key=lambda x: x[1].therapeutic_relevance)[0] if themes else None
            }
            
        except Exception as e:
            print(f"❌ Error creating summary statistics: {e}")
            return {}

def main():
    """Main execution function"""
    print("🔍 Conversation Topic and Theme Analysis System")
    print("=" * 55)
    
    # Initialize analyzer
    analyzer = TopicThemeAnalyzer()
    
    # Analyze topics and themes
    analysis_results = analyzer.analyze_topics_and_themes()
    
    if not analysis_results:
        print("❌ No analysis results generated")
        return
    
    # Export report
    report_file = analyzer.export_topic_theme_report(analysis_results)
    
    # Display summary
    topics = analysis_results.get('topics', {})
    themes = analysis_results.get('themes', {})
    insights = analysis_results.get('insights', [])
    
    print(f"\n✅ Topic and Theme Analysis Complete")
    print(f"   - Topics identified: {len(topics)}")
    print(f"   - Themes identified: {len(themes)}")
    print(f"   - Conversations analyzed: {analysis_results.get('total_conversations_analyzed', 0)}")
    print(f"   - Insights generated: {len(insights)}")
    print(f"   - Report saved: {report_file}")
    
    # Show top topics
    if topics:
        print(f"\n📊 Top Topics by Frequency:")
        sorted_topics = sorted(topics.items(), key=lambda x: x[1].frequency, reverse=True)
        for name, analysis in sorted_topics[:5]:
            print(f"   • {name.replace('_', ' ').title()}: {analysis.frequency} conversations ({analysis.percentage:.1f}%)")
    
    # Show themes
    if themes:
        print(f"\n🎭 Therapeutic Themes:")
        for name, analysis in themes.items():
            print(f"   • {name.replace('_', ' ').title()}: {analysis.conversation_count} conversations")
    
    # Show key insights
    if insights:
        print(f"\n🔍 Key Insights:")
        for insight in insights[:3]:
            print(f"   • {insight}")

if __name__ == "__main__":
    main()
