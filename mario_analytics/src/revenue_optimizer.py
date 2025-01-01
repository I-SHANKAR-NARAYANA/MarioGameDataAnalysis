import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class RevenueOptimizer:
    def __init__(self):
        self.pricing_models = {}
        self.offer_history = {}
        
    def calculate_price_elasticity(self, historical_data):
        """Calculate how sensitive players are to price changes"""
        df = pd.DataFrame(historical_data)
        
        # Simple price elasticity calculation
        price_changes = df['price'].pct_change()
        demand_changes = df['purchases'].pct_change()
        
        elasticity = (demand_changes / price_changes).mean()
        return elasticity
        
    def optimize_powerup_pricing(self, player_segment, current_price):
        """Dynamic pricing for in-game purchases"""
        
        segment_multipliers = {
            'Casual Explorer': 0.8,
            'Hardcore Completionist': 1.2,
            'Social Gamer': 1.0,
            'Speedrunner': 1.1,
            'Whale Spender': 1.5
        }
        
        base_multiplier = segment_multipliers.get(player_segment, 1.0)
        
        # Time-based adjustments
        hour = datetime.now().hour
        if 18 <= hour <= 22:  # Peak gaming hours
            time_multiplier = 1.1
        elif 6 <= hour <= 9:   # Morning commute
            time_multiplier = 0.9
        else:
            time_multiplier = 1.0
            
        optimized_price = current_price * base_multiplier * time_multiplier
        return round(optimized_price, 2)
        
    def generate_personalized_offers(self, player_data):
        """Create targeted offers based on player behavior"""
        
        offers = []
        
        # Struggling player - offer help
        if player_data.get('recent_deaths', 0) > 10:
            offers.append({
                'type': 'powerup_bundle',
                'description': 'Boost Pack - Get through tough levels!',
                'items': ['invincibility_star', 'fire_flower', 'super_mushroom'],
                'original_price': 4.99,
                'discounted_price': 2.99,
                'urgency': 'Limited time - 2 hours left!'
            })
            
        # High engagement player - premium content
        if player_data.get('session_length_avg', 0) > 30:
            offers.append({
                'type': 'premium_levels',
                'description': 'Exclusive Mario Worlds - For True Fans!',
                'items': ['world_9', 'world_10', 'bonus_challenges'],
                'price': 9.99,
                'value_proposition': 'Double your gameplay content'
            })
            
        # Social player - multiplayer features
        if player_data.get('multiplayer_sessions', 0) > 5:
            offers.append({
                'type': 'social_features',
                'description': 'Multiplayer Pro - Enhanced Co-op Experience',
                'items': ['custom_levels', 'player_chat', 'leaderboards'],
                'price': 6.99
            })
            
        return offers
        
    def calculate_offer_effectiveness(self, offer_id, conversion_data):
        """Measure ROI and effectiveness of offers"""
        
        total_shown = conversion_data.get('impressions', 0)
        total_converted = conversion_data.get('conversions', 0)
        total_revenue = conversion_data.get('revenue', 0)
        
        if total_shown == 0:
            return {'conversion_rate': 0, 'rpu': 0, 'roi': 0}
            
        conversion_rate = total_converted / total_shown
        revenue_per_user = total_revenue / total_shown
        
        return {
            'conversion_rate': conversion_rate,
            'revenue_per_user': revenue_per_user,
            'total_revenue': total_revenue,
            'effectiveness_score': conversion_rate * revenue_per_user
        }
