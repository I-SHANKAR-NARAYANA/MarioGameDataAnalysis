import numpy as np
from sklearn.ensemble import RandomForestRegressor
import json

class DynamicDifficultyAdjuster:
    def __init__(self):
        self.engagement_model = RandomForestRegressor(n_estimators=50)
        self.optimal_difficulty = 0.7  # Target difficulty sweet spot
        
    def calculate_current_difficulty(self, level_data):
        metrics = {
            'enemy_density': len(level_data.get('enemies', [])) / level_data.get('level_length', 1),
            'platform_complexity': self.analyze_platform_difficulty(level_data.get('platforms', [])),
            'powerup_availability': len(level_data.get('powerups', [])) / level_data.get('level_length', 1),
            'time_pressure': level_data.get('time_limit', 400) / 400,
        }
        return sum(metrics.values()) / len(metrics)
        
    def analyze_platform_difficulty(self, platforms):
        if not platforms:
            return 0.5
            
        gap_difficulty = sum(p.get('gap_size', 0) for p in platforms) / len(platforms) / 100
        moving_platforms = sum(1 for p in platforms if p.get('moving', False)) / len(platforms)
        
        return (gap_difficulty + moving_platforms) / 2
        
    def suggest_adjustments(self, player_performance, current_difficulty):
        performance_score = self.calculate_performance_score(player_performance)
        
        if performance_score < 0.3:  # Player struggling
            return {
                'enemy_reduction': 0.2,
                'additional_powerups': 2,
                'platform_simplification': 0.15,
                'time_bonus': 30
            }
        elif performance_score > 0.8:  # Player finding it too easy
            return {
                'enemy_increase': 0.15,
                'powerup_reduction': 1,
                'platform_complexity': 0.1,
                'time_reduction': 20
            }
        return {}
        
    def calculate_performance_score(self, performance):
        weights = {
            'completion_time': 0.3,
            'deaths': -0.4,
            'coins_collected_ratio': 0.2,
            'damage_taken': -0.1
        }
        
        score = 0
        for metric, weight in weights.items():
            score += performance.get(metric, 0) * weight
            
        return max(0, min(1, score))
