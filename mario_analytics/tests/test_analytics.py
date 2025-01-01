import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from player_behavior_model import PlayerBehaviorAnalyzer
from ab_testing import ABTestingFramework
from revenue_optimizer import RevenueOptimizer

class TestMarioAnalytics(unittest.TestCase):
    def setUp(self):
        self.analyzer = PlayerBehaviorAnalyzer()
        self.ab_tester = ABTestingFramework()
        self.revenue_optimizer = RevenueOptimizer()
        
    def test_player_behavior_features(self):
        mock_player_data = {
            'session_lengths': [30, 45, 20, 60],
            'levels_completed': 15,
            'levels_attempted': 20,
            'total_deaths': 45,
            'total_playtime': 155,
            'coins_collected': 1200,
            'powerups_used': 8,
            'powerups_found': 12,
            'multiplayer_sessions': 3,
            'iap_count': 2,
            'days_since_last_session': 1
        }
        
        features = self.analyzer.prepare_features(mock_player_data)
        self.assertIsNotNone(features)
        self.assertEqual(len(features.columns), 9)
        
    def test_ab_testing_assignment(self):
        experiment = self.ab_tester.create_experiment(
            'test_experiment', 
            ['control', 'variant_a', 'variant_b']
        )
        
        variant = self.ab_tester.assign_variant('player_123', 'test_experiment')
        self.assertIn(variant, ['control', 'variant_a', 'variant_b'])
        
        # Same player should get same variant
        variant2 = self.ab_tester.assign_variant('player_123', 'test_experiment')
        self.assertEqual(variant, variant2)
        
    def test_revenue_optimization(self):
        optimized_price = self.revenue_optimizer.optimize_powerup_pricing(
            'Whale Spender', 1.99
        )
        self.assertGreater(optimized_price, 1.99)
        
        offers = self.revenue_optimizer.generate_personalized_offers({
            'recent_deaths': 15,
            'session_length_avg': 45,
            'multiplayer_sessions': 8
        })
        
        self.assertGreater(len(offers), 0)
        
if __name__ == '__main__':
    unittest.main()
