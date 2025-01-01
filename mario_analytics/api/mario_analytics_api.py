from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from player_behavior_model import PlayerBehaviorAnalyzer
from ab_testing import ABTestingFramework
from revenue_optimizer import RevenueOptimizer

app = Flask(__name__)
CORS(app)

# Initialize components
behavior_analyzer = PlayerBehaviorAnalyzer()
ab_tester = ABTestingFramework()
revenue_optimizer = RevenueOptimizer()

@app.route('/api/player/analyze', methods=['POST'])
def analyze_player():
    player_data = request.json
    
    try:
        features = behavior_analyzer.prepare_features(player_data)
        
        results = {
            'churn_probability': float(behavior_analyzer.predict_churn_probability(features)),
            'lifetime_value': float(behavior_analyzer.predict_lifetime_value(features)),
            'player_segment': behavior_analyzer.segment_player(features),
            'recommended_actions': []
        }
        
        # Add recommendations based on analysis
        if results['churn_probability'] > 0.7:
            results['recommended_actions'].append('Send retention offer')
            results['recommended_actions'].append('Reduce difficulty temporarily')
            
        if results['player_segment'] == 'Whale Spender':
            results['recommended_actions'].append('Show premium content offers')
            
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/experiment/assign/<experiment_name>/<player_id>', methods=['GET'])
def assign_experiment(experiment_name, player_id):
    variant = ab_tester.assign_variant(player_id, experiment_name)
    
    if variant:
        return jsonify({'variant': variant, 'experiment': experiment_name})
    else:
        return jsonify({'error': 'Experiment not found'}), 404

@app.route('/api/revenue/optimize-price', methods=['POST'])
def optimize_pricing():
    data = request.json
    player_segment = data.get('player_segment')
    current_price = data.get('current_price', 1.99)
    
    optimized_price = revenue_optimizer.optimize_powerup_pricing(player_segment, current_price)
    
    return jsonify({
        'original_price': current_price,
        'optimized_price': optimized_price,
        'adjustment_factor': round(optimized_price / current_price, 2)
    })

@app.route('/api/offers/personalized/<player_id>', methods=['GET'])
def get_personalized_offers(player_id):
    # In real implementation, fetch player data from database
    mock_player_data = {
        'recent_deaths': 15,
        'session_length_avg': 45,
        'multiplayer_sessions': 8
    }
    
    offers = revenue_optimizer.generate_personalized_offers(mock_player_data)
    
    return jsonify({'player_id': player_id, 'offers': offers})

@app.route('/api/dashboard/metrics', methods=['GET'])
def get_dashboard_metrics():
    # Mock real-time metrics for dashboard
    metrics = {
        'active_players': 12847,
        'revenue_today': 23456.78,
        'avg_session_length': 28.5,
        'churn_rate_7d': 0.23,
        'top_performing_levels': [
            {'level': '1-1', 'completion_rate': 0.94},
            {'level': '1-2', 'completion_rate': 0.87},
            {'level': '2-1', 'completion_rate': 0.72}
        ],
        'ab_test_results': {
            'coin_reward_multiplier': {
                '1x': {'conversion_rate': 0.12},
                '1.5x': {'conversion_rate': 0.18},
                '2x': {'conversion_rate': 0.15}
            }
        }
    }
    
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
