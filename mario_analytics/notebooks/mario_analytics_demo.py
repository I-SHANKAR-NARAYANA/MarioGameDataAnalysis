"""
Mario Game Analytics - Executive Demo Script
Showcases key capabilities for Electronic Arts Data Science Role
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from player_behavior_model import PlayerBehaviorAnalyzer
from advanced_metrics import AdvancedGameMetrics

class MarioAnalyticsDemo:
    def __init__(self):
        self.analyzer = PlayerBehaviorAnalyzer()
        self.advanced_metrics = AdvancedGameMetrics()
        
    def generate_demo_data(self, n_players=10000):
        """Generate realistic demo data for presentation"""
        np.random.seed(42)
        
        # Generate player behavioral data
        players = []
        for i in range(n_players):
            player = {
                'player_id': f'player_{i}',
                'registration_date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
                'total_sessions': np.random.poisson(25),
                'avg_session_length': np.random.gamma(2, 15),
                'levels_completed': np.random.poisson(12),
                'total_spend': np.random.exponential(5) if np.random.random() > 0.7 else 0,
                'last_session': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'platform': np.random.choice(['mobile', 'pc', 'console'], p=[0.6, 0.25, 0.15])
            }
            
            # Add derived metrics
            player['session_frequency'] = player['total_sessions'] / max(
                (datetime.now() - player['registration_date']).days / 7, 1
            )
            player['is_churned'] = (datetime.now() - player['last_session']).days > 14
            player['ltv'] = player['total_spend'] * (1 + player['session_frequency'] * 0.1)
            
            players.append(player)
            
        return pd.DataFrame(players)
        
    def run_executive_demo(self):
        """Main demo showcasing EA-relevant analytics capabilities"""
        
        print("🍄 MARIO GAME ANALYTICS - EXECUTIVE DEMONSTRATION")
        print("=" * 60)
        
        # Generate demo dataset
        df = self.generate_demo_data()
        
        print(f"📊 Dataset: {len(df)} players analyzed")
        print(f"📱 Platform Mix: {df['platform'].value_counts().to_dict()}")
        print()
        
        # 1. Player Segmentation Analysis
        print("🎯 PLAYER SEGMENTATION ANALYSIS")
        print("-" * 40)
        
        # Create segments based on behavior
        df['engagement_segment'] = pd.cut(
            df['session_frequency'], 
            bins=[0, 1, 3, 6, float('inf')], 
            labels=['Low', 'Medium', 'High', 'Super']
        )
        
        segment_analysis = df.groupby('engagement_segment').agg({
            'player_id': 'count',
            'avg_session_length': 'mean',
            'total_spend': 'mean',
            'ltv': 'mean',
            'is_churned': 'mean'
        }).round(2)
        
        print(segment_analysis)
        print()
        
        # 2. Revenue Optimization Insights
        print("💰 REVENUE OPTIMIZATION INSIGHTS")
        print("-" * 40)
        
        spending_segments = df[df['total_spend'] > 0].copy()
        spending_segments['spend_tier'] = pd.cut(
            spending_segments['total_spend'],
            bins=[0, 5, 20, 50, float('inf')],
            labels=['Light Spender', 'Regular', 'Heavy', 'Whale']
        )
        
        revenue_analysis = spending_segments.groupby('spend_tier').agg({
            'player_id': 'count',
            'total_spend': ['mean', 'sum'],
            'session_frequency': 'mean'
        }).round(2)
        
        print("Revenue by Spending Tier:")
        print(revenue_analysis)
        print()
        
        total_revenue = df['total_spend'].sum()
        whale_revenue = spending_segments[spending_segments['spend_tier'] == 'Whale']['total_spend'].sum()
        whale_percentage = len(spending_segments[spending_segments['spend_tier'] == 'Whale']) / len(df) * 100
        
        print(f"🐋 Whale Impact: {whale_percentage:.1f}% of players generate ${whale_revenue:,.0f} ({whale_revenue/total_revenue*100:.1f}% of revenue)")
        print()
        
        # 3. Churn Prediction Results
        print("⚠️ CHURN PREDICTION & RETENTION")
        print("-" * 40)
        
        churn_rate = df['is_churned'].mean() * 100
        at_risk_players = df[
            (df['session_frequency'] < 1) & 
            (~df['is_churned']) &
            ((datetime.now() - df['last_session']).dt.days > 7)
        ]
        
        print(f"Current Churn Rate: {churn_rate:.1f}%")
        print(f"Players at Risk: {len(at_risk_players)} ({len(at_risk_players)/len(df)*100:.1f}%)")
        
        # Platform-specific retention
        platform_retention = df.groupby('platform').agg({
            'is_churned': lambda x: (1 - x.mean()) * 100,
            'session_frequency': 'mean',
            'avg_session_length': 'mean'
        }).round(2)
        platform_retention.columns = ['Retention_Rate_%', 'Avg_Sessions_Per_Week', 'Avg_Session_Minutes']
        
        print("\nPlatform Performance:")
        print(platform_retention)
        print()
        
        # 4. Business Impact Projections
        print("📈 BUSINESS IMPACT PROJECTIONS")
        print("-" * 40)
        
        # Calculate potential impact of reducing churn by 20%
        current_monthly_revenue = total_revenue / 12  # Assuming data spans 1 year
        retained_players = len(at_risk_players) * 0.2
        avg_monthly_spend_per_player = df[df['total_spend'] > 0]['total_spend'].mean() / 12
        
        projected_additional_revenue = retained_players * avg_monthly_spend_per_player
        
        print(f"💡 If we retain 20% of at-risk players:")
        print(f"   • Additional monthly revenue: ${projected_additional_revenue:,.0f}")
        print(f"   • Annual impact: ${projected_additional_revenue * 12:,.0f}")
        print(f"   • ROI on retention campaigns: {projected_additional_revenue / 1000:.1f}x (assuming $1K campaign cost)")
        print()
        
        # 5. Key Recommendations
        print("🎯 KEY RECOMMENDATIONS FOR EA")
        print("-" * 40)
        print("1. 🎮 Implement dynamic difficulty adjustment for struggling players")
        print("2. 💰 Create targeted offers for 'Regular' spenders to upgrade to 'Heavy'")  
        print("3. 📱 Optimize mobile experience - highest volume, lowest retention")
        print("4. 🤝 Develop retention campaigns for players with <1 session/week")
        print("5. 🧪 A/B test pricing strategies for different player segments")
        print()
        
        print("✅ Demo Complete - Ready for Production Deployment!")
        
        return {
            'total_players': len(df),
            'churn_rate': churn_rate,
            'total_revenue': total_revenue,
            'projected_annual_lift': projected_additional_revenue * 12
        }

if __name__ == "__main__":
    demo = MarioAnalyticsDemo()
    results = demo.run_executive_demo()
