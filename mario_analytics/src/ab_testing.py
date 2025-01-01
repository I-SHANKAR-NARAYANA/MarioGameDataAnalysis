import random
import hashlib
import pandas as pd
from datetime import datetime, timedelta
import json

class ABTestingFramework:
    def __init__(self):
        self.active_experiments = {}
        self.results_cache = {}
        
    def create_experiment(self, experiment_name, variants, traffic_allocation=None):
        if traffic_allocation is None:
            traffic_allocation = {variant: 1.0/len(variants) for variant in variants}
            
        experiment = {
            'name': experiment_name,
            'variants': variants,
            'traffic_allocation': traffic_allocation,
            'start_date': datetime.utcnow(),
            'status': 'active',
            'participants': {},
            'conversion_data': {variant: [] for variant in variants}
        }
        
        self.active_experiments[experiment_name] = experiment
        return experiment
        
    def assign_variant(self, player_id, experiment_name):
        if experiment_name not in self.active_experiments:
            return None
            
        # Consistent hashing for stable assignment
        hash_input = f"{player_id}_{experiment_name}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        assignment_value = (hash_value % 10000) / 10000
        
        experiment = self.active_experiments[experiment_name]
        cumulative_allocation = 0
        
        for variant, allocation in experiment['traffic_allocation'].items():
            cumulative_allocation += allocation
            if assignment_value <= cumulative_allocation:
                experiment['participants'][player_id] = variant
                return variant
                
        return list(experiment['variants'])[0]  # Fallback
        
    def track_conversion(self, player_id, experiment_name, conversion_value):
        if experiment_name not in self.active_experiments:
            return False
            
        experiment = self.active_experiments[experiment_name]
        variant = experiment['participants'].get(player_id)
        
        if variant:
            experiment['conversion_data'][variant].append({
                'player_id': player_id,
                'conversion_value': conversion_value,
                'timestamp': datetime.utcnow()
            })
            return True
        return False
        
    def analyze_experiment_results(self, experiment_name):
        if experiment_name not in self.active_experiments:
            return None
            
        experiment = self.active_experiments[experiment_name]
        results = {}
        
        for variant, conversions in experiment['conversion_data'].items():
            if conversions:
                conversion_rate = len(conversions) / len(experiment['participants'])
                avg_value = sum(c['conversion_value'] for c in conversions) / len(conversions)
                
                results[variant] = {
                    'participants': len([p for p, v in experiment['participants'].items() if v == variant]),
                    'conversions': len(conversions),
                    'conversion_rate': conversion_rate,
                    'avg_conversion_value': avg_value
                }
            else:
                results[variant] = {
                    'participants': len([p for p, v in experiment['participants'].items() if v == variant]),
                    'conversions': 0,
                    'conversion_rate': 0.0,
                    'avg_conversion_value': 0.0
                }
                
        return results

# Pre-configured experiments for Mario game
MARIO_EXPERIMENTS = {
    'coin_reward_multiplier': {
        'variants': ['1x', '1.5x', '2x'],
        'hypothesis': 'Higher coin rewards increase engagement'
    },
    'powerup_spawn_rate': {
        'variants': ['low', 'medium', 'high'],
        'hypothesis': 'Optimal powerup frequency maximizes retention'
    },
    'level_art_style': {
        'variants': ['classic', 'modern', 'neon'],
        'hypothesis': 'Art style affects player preference and session length'
    }
}
