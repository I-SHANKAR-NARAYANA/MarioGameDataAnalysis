import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class AdvancedGameMetrics:
    def __init__(self):
        self.cohort_data = {}
        self.funnel_data = {}
        
    def calculate_player_cohorts(self, user_data):
        """Cohort analysis for retention tracking"""
        df = pd.DataFrame(user_data)
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        df['activity_date'] = pd.to_datetime(df['activity_date'])
        
        # Create cohort month
        df['cohort_month'] = df['registration_date'].dt.to_period('M')
        df['activity_month'] = df['activity_date'].dt.to_period('M')
        
        # Calculate period number
        df['period_number'] = (df['activity_month'] - df['cohort_month']).apply(attrgetter('n'))
        
        cohort_table = df.pivot_table(
            index='cohort_month',
            columns='period_number',
            values='player_id',
            aggfunc='nunique'
        ).fillna(0)
        
        # Calculate retention rates
        cohort_sizes = df.groupby('cohort_month')['player_id'].nunique()
        retention_table = cohort_table.divide(cohort_sizes, axis=0)
        
        return retention_table
        
    def analyze_level_funnel(self, level_progression_data):
        """Analyze where players drop off in level progression"""
        df = pd.DataFrame(level_progression_data)
        
        funnel_analysis = {}
        levels = sorted(df['level'].unique())
        
        for i, level in enumerate(levels):
            players_at_level = df[df['level'] >= level]['player_id'].nunique()
            
            if i == 0:
                drop_off_rate = 0
            else:
                previous_level_players = df[df['level'] >= levels[i-1]]['player_id'].nunique()
                drop_off_rate = 1 - (players_at_level / previous_level_players)
                
            funnel_analysis[level] = {
                'players_reached': players_at_level,
                'drop_off_rate': drop_off_rate,
                'completion_rate': df[df['level'] == level]['completed'].mean() if level in df['level'].values else 0
            }
            
        return funnel_analysis
        
    def calculate_engagement_score(self, player_session_data):
        """Multi-dimensional engagement scoring"""
        
        weights = {
            'session_frequency': 0.25,    # How often they play
            'session_duration': 0.20,     # How long they play
            'feature_adoption': 0.15,     # Use of game features
            'social_interaction': 0.15,   # Multiplayer engagement
            'progression_rate': 0.15,     # Level completion speed
            'monetization': 0.10          # IAP behavior
        }
        
        scores = {}
        for player_id, data in player_session_data.items():
            
            # Normalize each metric to 0-1 scale
            session_freq_score = min(data.get('sessions_per_week', 0) / 10, 1)
            duration_score = min(data.get('avg_session_minutes', 0) / 60, 1)
            feature_score = data.get('features_used', 0) / 10  # Assuming 10 total features
            social_score = min(data.get('multiplayer_sessions', 0) / 5, 1)
            progression_score = data.get('levels_completed_per_hour', 0) / 3
            monetization_score = min(data.get('lifetime_spend', 0) / 50, 1)
            
            engagement_score = (
                session_freq_score * weights['session_frequency'] +
                duration_score * weights['session_duration'] +
                feature_score * weights['feature_adoption'] +
                social_score * weights['social_interaction'] +
                progression_score * weights['progression_rate'] +
                monetization_score * weights['monetization']
            )
            
            scores[player_id] = {
                'overall_score': engagement_score,
                'segment': self.classify_engagement_segment(engagement_score),
                'breakdown': {
                    'frequency': session_freq_score,
                    'duration': duration_score,
                    'features': feature_score,
                    'social': social_score,
                    'progression': progression_score,
                    'monetization': monetization_score
                }
            }
            
        return scores
        
    def classify_engagement_segment(self, score):
        if score >= 0.8:
            return "Super Engaged"
        elif score >= 0.6:
            return "Highly Engaged"  
        elif score >= 0.4:
            return "Moderately Engaged"
        elif score >= 0.2:
            return "Low Engagement"
        else:
            return "At Risk"
