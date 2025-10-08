#!/usr/bin/env python3
"""
Conversation Effectiveness Predictor
Task 5.6.3.8: Implement conversation effectiveness prediction

Predicts conversation effectiveness using machine learning models:
- Feature extraction from conversation characteristics
- Effectiveness scoring based on multiple criteria
- Predictive modeling for conversation outcomes
- Performance metrics and validation
- Effectiveness improvement recommendations
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease, flesch_kincaid_grade
import warnings
warnings.filterwarnings('ignore')

class ConversationEffectivenessPredictor:
    def __init__(self, db_path: str = "/home/vivi/pixelated/ai/database/conversations.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
        
    def connect_db(self):
        """Connect to the conversations database"""
        return sqlite3.connect(self.db_path)
    
    def predict_effectiveness(self) -> Dict[str, Any]:
        """Main function for conversation effectiveness prediction"""
        print("🎯 Starting Conversation Effectiveness Prediction...")
        
        # Load and prepare data
        conversations = self._load_conversation_data()
        print(f"📊 Loaded {len(conversations)} conversations for analysis")
        
        # Extract features
        features_df = self._extract_effectiveness_features(conversations)
        print(f"🔧 Extracted {len(features_df.columns)} features")
        
        # Calculate effectiveness scores
        effectiveness_scores = self._calculate_effectiveness_scores(conversations)
        
        # Combine features with effectiveness scores
        modeling_data = features_df.join(effectiveness_scores, how='inner')
        
        # Build predictive models
        model_results = self._build_prediction_models(modeling_data)
        
        # Generate predictions for all conversations
        predictions = self._generate_predictions(modeling_data, model_results)
        
        # Analyze effectiveness patterns
        effectiveness_analysis = self._analyze_effectiveness_patterns(modeling_data, predictions)
        
        # Create visualizations
        self._create_effectiveness_visualizations(modeling_data, predictions, model_results)
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_conversations': len(conversations),
            'feature_count': len(features_df.columns),
            'model_performance': model_results,
            'effectiveness_analysis': effectiveness_analysis,
            'predictions': predictions,
            'insights': self._generate_effectiveness_insights(effectiveness_analysis, model_results),
            'recommendations': self._generate_effectiveness_recommendations(effectiveness_analysis)
        }
    
    def _load_conversation_data(self) -> pd.DataFrame:
        """Load conversation data for effectiveness prediction"""
        with self.connect_db() as conn:
            query = """
            SELECT 
                conversation_id, dataset_source as dataset, tier, conversations_json, 
                character_count as text_length,
                word_count,
                turn_count,
                created_at
            FROM conversations 
            WHERE conversations_json IS NOT NULL 
            AND length(conversations_json) > 10
            """
            df = pd.read_sql_query(query, conn)
            
        # Extract conversation text from JSON
        df['conversation_text'] = df['conversations_json'].apply(self._extract_text_from_json)
        
        # Filter out empty conversations
        df = df[df['conversation_text'].str.len() > 10]
        
        return df
    
    def _extract_text_from_json(self, json_str: str) -> str:
        """Extract readable text from conversations JSON"""
        try:
            import json
            conversations = json.loads(json_str)
            
            if isinstance(conversations, list):
                text_parts = []
                for turn in conversations:
                    if isinstance(turn, dict):
                        for role, content in turn.items():
                            text_parts.append(f"{role}: {content}")
                    else:
                        text_parts.append(str(turn))
                return '\n'.join(text_parts)
            else:
                return str(conversations)
        except:
            return json_str
    
    def _extract_effectiveness_features(self, conversations: pd.DataFrame) -> pd.DataFrame:
        """Extract features that correlate with conversation effectiveness"""
        print("🔧 Extracting effectiveness features...")
        
        features = []
        
        for _, conv in conversations.iterrows():
            text = conv['conversation_text']
            
            # Basic metrics
            text_length = len(text)
            word_count = len(text.split())
            sentence_count = len([s for s in text.split('.') if s.strip()])
            
            # Readability features
            try:
                flesch_score = flesch_reading_ease(text)
                fk_grade = flesch_kincaid_grade(text)
            except:
                flesch_score = 50
                fk_grade = 8
            
            # Engagement features
            question_count = text.count('?')
            exclamation_count = text.count('!')
            dialogue_turns = conv['turn_count']
            
            # Emotional features
            positive_words = len(re.findall(r'\b(good|great|excellent|wonderful|amazing|helpful|thank|appreciate|love|happy|joy)\b', text.lower()))
            negative_words = len(re.findall(r'\b(bad|terrible|awful|horrible|hate|angry|frustrated|disappointed|sad|upset)\b', text.lower()))
            empathy_words = len(re.findall(r'\b(understand|feel|sorry|empathize|support|care|concern|comfort)\b', text.lower()))
            
            # Interaction features
            personal_pronouns = len(re.findall(r'\b(I|you|we|us|your|my|our)\b', text.lower()))
            conversational_markers = len(re.findall(r'\b(well|so|now|then|actually|really|you know|I mean|let me|how about)\b', text.lower()))
            
            # Structure features
            has_lists = bool(re.search(r'\n\s*[-*•]\s+', text))
            has_numbered_items = bool(re.search(r'\n\s*\d+\.\s+', text))
            has_questions = question_count > 0
            
            # Complexity features
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            unique_words = len(set(text.lower().split()))
            vocabulary_richness = unique_words / word_count if word_count > 0 else 0
            
            # Topic coherence features
            transition_words = len(re.findall(r'\b(however|therefore|moreover|furthermore|additionally|meanwhile|consequently|thus|hence)\b', text.lower()))
            
            # Professional language features
            professional_terms = len(re.findall(r'\b(analysis|evaluation|assessment|methodology|implementation|optimization|strategy|solution)\b', text.lower()))
            
            # Dataset and tier encoding
            dataset_encoded = hash(conv['dataset']) % 1000  # Simple hash encoding
            tier_encoded = hash(conv['tier']) % 100
            
            feature_vector = {
                'conversation_id': conv['conversation_id'],
                'text_length': text_length,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'flesch_score': flesch_score,
                'fk_grade': fk_grade,
                'question_count': question_count,
                'exclamation_count': exclamation_count,
                'dialogue_turns': dialogue_turns,
                'positive_words': positive_words,
                'negative_words': negative_words,
                'empathy_words': empathy_words,
                'personal_pronouns': personal_pronouns,
                'conversational_markers': conversational_markers,
                'has_lists': int(has_lists),
                'has_numbered_items': int(has_numbered_items),
                'has_questions': int(has_questions),
                'avg_sentence_length': avg_sentence_length,
                'vocabulary_richness': vocabulary_richness,
                'transition_words': transition_words,
                'professional_terms': professional_terms,
                'dataset_encoded': dataset_encoded,
                'tier_encoded': tier_encoded,
                'question_density': question_count / word_count * 100 if word_count > 0 else 0,
                'empathy_density': empathy_words / word_count * 100 if word_count > 0 else 0,
                'sentiment_balance': (positive_words - negative_words) / word_count * 100 if word_count > 0 else 0
            }
            
            features.append(feature_vector)
        
        features_df = pd.DataFrame(features)
        features_df.set_index('conversation_id', inplace=True)
        
        return features_df
    
    def _calculate_effectiveness_scores(self, conversations: pd.DataFrame) -> pd.Series:
        """Calculate effectiveness scores based on multiple criteria"""
        print("📊 Calculating effectiveness scores...")
        
        effectiveness_scores = {}
        
        for _, conv in conversations.iterrows():
            text = conv['conversation_text']
            word_count = len(text.split())
            
            # Component scores (0-100 scale)
            
            # 1. Engagement score
            questions = text.count('?')
            exclamations = text.count('!')
            personal_pronouns = len(re.findall(r'\b(I|you|we|us|your|my|our)\b', text.lower()))
            engagement_score = min(100, (questions * 10) + (exclamations * 5) + (personal_pronouns / word_count * 200))
            
            # 2. Clarity score
            try:
                flesch_score = flesch_reading_ease(text)
                clarity_score = max(0, min(100, flesch_score))
            except:
                clarity_score = 50
            
            # 3. Empathy score
            empathy_words = len(re.findall(r'\b(understand|feel|sorry|empathize|support|care|concern|comfort)\b', text.lower()))
            empathy_score = min(100, empathy_words / word_count * 500)
            
            # 4. Informativeness score
            unique_words = len(set(text.lower().split()))
            vocabulary_richness = unique_words / word_count if word_count > 0 else 0
            professional_terms = len(re.findall(r'\b(analysis|evaluation|assessment|methodology|implementation|solution)\b', text.lower()))
            informativeness_score = min(100, (vocabulary_richness * 100) + (professional_terms * 5))
            
            # 5. Structure score
            has_lists = bool(re.search(r'\n\s*[-*•]\s+', text))
            has_numbered_items = bool(re.search(r'\n\s*\d+\.\s+', text))
            transition_words = len(re.findall(r'\b(however|therefore|moreover|furthermore|additionally)\b', text.lower()))
            structure_score = (int(has_lists) * 20) + (int(has_numbered_items) * 20) + (transition_words * 10)
            structure_score = min(100, structure_score)
            
            # 6. Responsiveness score (based on dialogue turns and length appropriateness)
            turn_count = conv['turn_count']
            optimal_length = 200 + (turn_count * 50)  # Adaptive optimal length
            length_penalty = abs(len(text) - optimal_length) / optimal_length
            responsiveness_score = max(0, 100 - (length_penalty * 50))
            
            # Weighted overall effectiveness score
            overall_effectiveness = (
                engagement_score * 0.20 +
                clarity_score * 0.20 +
                empathy_score * 0.15 +
                informativeness_score * 0.20 +
                structure_score * 0.15 +
                responsiveness_score * 0.10
            )
            
            effectiveness_scores[conv['conversation_id']] = overall_effectiveness
        
        return pd.Series(effectiveness_scores, name='effectiveness_score')
    
    def _build_prediction_models(self, modeling_data: pd.DataFrame) -> Dict[str, Any]:
        """Build machine learning models to predict effectiveness"""
        print("🤖 Building prediction models...")
        
        # Prepare features and target
        feature_columns = [col for col in modeling_data.columns if col != 'effectiveness_score']
        X = modeling_data[feature_columns]
        y = modeling_data['effectiveness_score']
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers['effectiveness'] = scaler
        
        # Train multiple models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        model_results = {}
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            # Train model
            if model_name == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            if model_name == 'linear_regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, model.feature_importances_))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            else:
                top_features = []
            
            model_results[model_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'top_features': top_features,
                'predictions': y_pred.tolist()[:100]  # Sample predictions
            }
            
            # Store best model
            if model_name == 'random_forest':  # Use Random Forest as primary model
                self.models['effectiveness'] = model
                self.feature_importance['effectiveness'] = feature_importance
        
        return model_results
    
    def _generate_predictions(self, modeling_data: pd.DataFrame, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate effectiveness predictions for all conversations"""
        print("🔮 Generating effectiveness predictions...")
        
        # Use the best performing model (Random Forest)
        model = self.models.get('effectiveness')
        if not model:
            return {'error': 'No trained model available'}
        
        # Prepare features
        feature_columns = [col for col in modeling_data.columns if col != 'effectiveness_score']
        X = modeling_data[feature_columns].fillna(modeling_data[feature_columns].mean())
        
        # Generate predictions
        predicted_effectiveness = model.predict(X)
        actual_effectiveness = modeling_data['effectiveness_score'].values
        
        # Calculate prediction accuracy
        prediction_error = np.abs(predicted_effectiveness - actual_effectiveness)
        
        # Categorize conversations by predicted effectiveness
        effectiveness_categories = []
        for score in predicted_effectiveness:
            if score >= 80:
                effectiveness_categories.append('highly_effective')
            elif score >= 60:
                effectiveness_categories.append('moderately_effective')
            elif score >= 40:
                effectiveness_categories.append('somewhat_effective')
            else:
                effectiveness_categories.append('needs_improvement')
        
        category_distribution = Counter(effectiveness_categories)
        
        return {
            'predicted_scores': predicted_effectiveness.tolist(),
            'actual_scores': actual_effectiveness.tolist(),
            'prediction_errors': prediction_error.tolist(),
            'mean_prediction_error': prediction_error.mean(),
            'category_distribution': dict(category_distribution),
            'high_effectiveness_conversations': len([s for s in predicted_effectiveness if s >= 80]),
            'low_effectiveness_conversations': len([s for s in predicted_effectiveness if s < 40])
        }
    
    def _analyze_effectiveness_patterns(self, modeling_data: pd.DataFrame, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in conversation effectiveness"""
        print("📈 Analyzing effectiveness patterns...")
        
        # Add predictions to modeling data for analysis
        modeling_data_with_pred = modeling_data.copy()
        modeling_data_with_pred['predicted_effectiveness'] = predictions['predicted_scores']
        
        patterns_analysis = {}
        
        # Effectiveness by feature ranges
        feature_effectiveness = {}
        key_features = ['text_length', 'dialogue_turns', 'empathy_density', 'question_density', 'flesch_score']
        
        for feature in key_features:
            if feature in modeling_data_with_pred.columns:
                # Create quartiles
                quartiles = modeling_data_with_pred[feature].quantile([0.25, 0.5, 0.75])
                
                q1_mask = modeling_data_with_pred[feature] <= quartiles[0.25]
                q2_mask = (modeling_data_with_pred[feature] > quartiles[0.25]) & (modeling_data_with_pred[feature] <= quartiles[0.5])
                q3_mask = (modeling_data_with_pred[feature] > quartiles[0.5]) & (modeling_data_with_pred[feature] <= quartiles[0.75])
                q4_mask = modeling_data_with_pred[feature] > quartiles[0.75]
                
                feature_effectiveness[feature] = {
                    'q1_effectiveness': modeling_data_with_pred[q1_mask]['effectiveness_score'].mean(),
                    'q2_effectiveness': modeling_data_with_pred[q2_mask]['effectiveness_score'].mean(),
                    'q3_effectiveness': modeling_data_with_pred[q3_mask]['effectiveness_score'].mean(),
                    'q4_effectiveness': modeling_data_with_pred[q4_mask]['effectiveness_score'].mean(),
                    'correlation': modeling_data_with_pred[feature].corr(modeling_data_with_pred['effectiveness_score'])
                }
        
        patterns_analysis['feature_effectiveness'] = feature_effectiveness
        
        # Most and least effective conversation characteristics
        top_10_effective = modeling_data_with_pred.nlargest(10, 'effectiveness_score')
        bottom_10_effective = modeling_data_with_pred.nsmallest(10, 'effectiveness_score')
        
        patterns_analysis['top_effective_characteristics'] = {
            'avg_text_length': top_10_effective['text_length'].mean(),
            'avg_dialogue_turns': top_10_effective['dialogue_turns'].mean(),
            'avg_empathy_density': top_10_effective['empathy_density'].mean(),
            'avg_question_density': top_10_effective['question_density'].mean(),
            'avg_flesch_score': top_10_effective['flesch_score'].mean()
        }
        
        patterns_analysis['bottom_effective_characteristics'] = {
            'avg_text_length': bottom_10_effective['text_length'].mean(),
            'avg_dialogue_turns': bottom_10_effective['dialogue_turns'].mean(),
            'avg_empathy_density': bottom_10_effective['empathy_density'].mean(),
            'avg_question_density': bottom_10_effective['question_density'].mean(),
            'avg_flesch_score': bottom_10_effective['flesch_score'].mean()
        }
        
        # Effectiveness distribution analysis
        effectiveness_stats = {
            'mean_effectiveness': modeling_data_with_pred['effectiveness_score'].mean(),
            'median_effectiveness': modeling_data_with_pred['effectiveness_score'].median(),
            'std_effectiveness': modeling_data_with_pred['effectiveness_score'].std(),
            'min_effectiveness': modeling_data_with_pred['effectiveness_score'].min(),
            'max_effectiveness': modeling_data_with_pred['effectiveness_score'].max()
        }
        
        patterns_analysis['effectiveness_statistics'] = effectiveness_stats
        
        return patterns_analysis
    
    def _generate_effectiveness_insights(self, effectiveness_analysis: Dict[str, Any], model_results: Dict[str, Any]) -> List[str]:
        """Generate insights from effectiveness analysis"""
        insights = []
        
        # Model performance insights
        best_model = max(model_results.items(), key=lambda x: x[1]['r2_score'])
        insights.append(f"🤖 Best performing model: {best_model[0]} (R² = {best_model[1]['r2_score']:.3f})")
        
        # Feature importance insights
        if 'random_forest' in model_results and model_results['random_forest']['top_features']:
            top_feature = model_results['random_forest']['top_features'][0]
            insights.append(f"🔑 Most important effectiveness predictor: {top_feature[0]} (importance: {top_feature[1]:.3f})")
        
        # Effectiveness distribution insights
        stats = effectiveness_analysis['effectiveness_statistics']
        if stats['mean_effectiveness'] < 50:
            insights.append(f"⚠️ Average effectiveness is low ({stats['mean_effectiveness']:.1f}) - significant improvement needed")
        elif stats['mean_effectiveness'] > 70:
            insights.append(f"✅ Good average effectiveness ({stats['mean_effectiveness']:.1f}) - maintain current quality")
        
        # Feature correlation insights
        feature_eff = effectiveness_analysis['feature_effectiveness']
        strong_correlations = [(feature, data['correlation']) for feature, data in feature_eff.items() 
                              if abs(data['correlation']) > 0.3]
        
        if strong_correlations:
            strongest = max(strong_correlations, key=lambda x: abs(x[1]))
            insights.append(f"📊 Strongest effectiveness correlation: {strongest[0]} (r = {strongest[1]:.3f})")
        
        return insights
    
    def _generate_effectiveness_recommendations(self, effectiveness_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving effectiveness"""
        recommendations = []
        
        # Feature-based recommendations
        feature_eff = effectiveness_analysis['feature_effectiveness']
        
        # Text length recommendations
        if 'text_length' in feature_eff:
            length_data = feature_eff['text_length']
            best_quartile = max([(f'q{i+1}', length_data[f'q{i+1}_effectiveness']) for i in range(4)], key=lambda x: x[1])
            if best_quartile[0] == 'q2' or best_quartile[0] == 'q3':
                recommendations.append("📝 Optimize conversation length - moderate lengths tend to be most effective")
        
        # Empathy recommendations
        if 'empathy_density' in feature_eff:
            empathy_corr = feature_eff['empathy_density']['correlation']
            if empathy_corr > 0.2:
                recommendations.append("❤️ Increase empathetic language - strong positive correlation with effectiveness")
        
        # Question density recommendations
        if 'question_density' in feature_eff:
            question_corr = feature_eff['question_density']['correlation']
            if question_corr > 0.15:
                recommendations.append("❓ Include more questions - enhances conversation effectiveness")
        
        # Readability recommendations
        if 'flesch_score' in feature_eff:
            flesch_corr = feature_eff['flesch_score']['correlation']
            if flesch_corr > 0.1:
                recommendations.append("📖 Improve readability - clearer language increases effectiveness")
        
        # Dialogue turn recommendations
        if 'dialogue_turns' in feature_eff:
            turns_corr = feature_eff['dialogue_turns']['correlation']
            if turns_corr > 0.1:
                recommendations.append("💬 Encourage more dialogue turns - interactive conversations are more effective")
        
        # General recommendations based on effectiveness statistics
        stats = effectiveness_analysis['effectiveness_statistics']
        if stats['std_effectiveness'] > 20:
            recommendations.append("⚖️ Reduce effectiveness variability - standardize conversation quality")
        
        return recommendations
    
    def _create_effectiveness_visualizations(self, modeling_data: pd.DataFrame, predictions: Dict[str, Any], model_results: Dict[str, Any]):
        """Create visualizations for effectiveness analysis"""
        print("📊 Creating effectiveness visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Conversation Effectiveness Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Effectiveness Score Distribution
        effectiveness_scores = modeling_data['effectiveness_score']
        
        axes[0, 0].hist(effectiveness_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Effectiveness Score Distribution')
        axes[0, 0].set_xlabel('Effectiveness Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(effectiveness_scores.mean(), color='red', linestyle='--', 
                          label=f'Mean: {effectiveness_scores.mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Model Performance Comparison
        models = list(model_results.keys())
        r2_scores = [model_results[model]['r2_score'] for model in models]
        
        bars = axes[0, 1].bar(models, r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        axes[0, 1].set_title('Model Performance (R² Score)')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Predicted vs Actual Effectiveness
        if 'predicted_scores' in predictions and 'actual_scores' in predictions:
            predicted = predictions['predicted_scores']
            actual = predictions['actual_scores']
            
            axes[0, 2].scatter(actual, predicted, alpha=0.6, s=20)
            axes[0, 2].plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', lw=2)
            axes[0, 2].set_title('Predicted vs Actual Effectiveness')
            axes[0, 2].set_xlabel('Actual Effectiveness')
            axes[0, 2].set_ylabel('Predicted Effectiveness')
            
            # Add R² annotation
            r2 = model_results.get('random_forest', {}).get('r2_score', 0)
            axes[0, 2].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 2].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Feature Importance (Random Forest)
        if 'random_forest' in model_results and model_results['random_forest']['top_features']:
            top_features = model_results['random_forest']['top_features'][:8]  # Top 8 features
            feature_names = [f[0] for f in top_features]
            importance_values = [f[1] for f in top_features]
            
            y_pos = np.arange(len(feature_names))
            bars = axes[1, 0].barh(y_pos, importance_values, color='lightcoral', alpha=0.8)
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels([name.replace('_', ' ').title() for name in feature_names])
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top Features for Effectiveness Prediction')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, importance_values)):
                width = bar.get_width()
                axes[1, 0].text(width + 0.001, bar.get_y() + bar.get_height()/2,
                               f'{value:.3f}', ha='left', va='center')
        
        # 5. Effectiveness Category Distribution
        if 'category_distribution' in predictions:
            categories = list(predictions['category_distribution'].keys())
            counts = list(predictions['category_distribution'].values())
            colors = ['#FF6B6B', '#FFA07A', '#98D8C8', '#87CEEB']
            
            wedges, texts, autotexts = axes[1, 1].pie(counts, labels=categories, autopct='%1.1f%%', 
                                                     colors=colors, startangle=90)
            axes[1, 1].set_title('Effectiveness Category Distribution')
            
            # Improve label readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 6. Effectiveness vs Key Features
        key_features = ['empathy_density', 'question_density', 'flesch_score']
        colors = ['red', 'blue', 'green']
        
        for i, (feature, color) in enumerate(zip(key_features, colors)):
            if feature in modeling_data.columns:
                x_vals = modeling_data[feature]
                y_vals = modeling_data['effectiveness_score']
                
                # Calculate correlation
                correlation = x_vals.corr(y_vals)
                
                axes[1, 2].scatter(x_vals, y_vals, alpha=0.5, s=15, color=color, 
                                  label=f'{feature.replace("_", " ").title()} (r={correlation:.2f})')
        
        axes[1, 2].set_xlabel('Feature Value')
        axes[1, 2].set_ylabel('Effectiveness Score')
        axes[1, 2].set_title('Effectiveness vs Key Features')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'/home/vivi/pixelated/ai/monitoring/effectiveness_prediction_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Effectiveness prediction visualizations saved as effectiveness_prediction_{timestamp}.png")

def main():
    """Main execution function"""
    print("🚀 Starting Conversation Effectiveness Prediction System")
    print("=" * 65)
    
    predictor = ConversationEffectivenessPredictor()
    
    try:
        # Run the complete prediction analysis
        results = predictor.predict_effectiveness()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/home/vivi/pixelated/ai/monitoring/effectiveness_prediction_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_file}")
        print(f"📊 Total conversations analyzed: {results['total_conversations']}")
        print(f"🔧 Features extracted: {results['feature_count']}")
        print(f"🎯 Generated {len(results['insights'])} insights")
        print(f"💡 Generated {len(results['recommendations'])} recommendations")
        
        # Display model performance
        print("\n🤖 Model Performance:")
        for model_name, metrics in results['model_performance'].items():
            print(f"  • {model_name}: R² = {metrics['r2_score']:.3f}, RMSE = {metrics['rmse']:.2f}")
        
        # Display key insights
        print("\n🎯 Key Effectiveness Insights:")
        for insight in results['insights'][:5]:
            print(f"  • {insight}")
        
        print("\n💡 Top Recommendations:")
        for rec in results['recommendations'][:3]:
            print(f"  • {rec}")
        
        # Display prediction summary
        predictions = results['predictions']
        print(f"\n📈 Prediction Summary:")
        print(f"  • High effectiveness conversations: {predictions['high_effectiveness_conversations']}")
        print(f"  • Low effectiveness conversations: {predictions['low_effectiveness_conversations']}")
        print(f"  • Mean prediction error: {predictions['mean_prediction_error']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
