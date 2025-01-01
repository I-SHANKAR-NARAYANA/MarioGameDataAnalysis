import json
import time
from datetime import datetime
import websocket
import pandas as pd

class MarioGameAnalytics:
    def __init__(self):
        self.events = []
        self.session_data = {}
        
    def track_event(self, event_type, player_id, data):
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'player_id': player_id,
            'session_id': self.session_data.get('session_id'),
            'data': data
        }
        self.events.append(event)
        
    def track_player_movement(self, x, y, velocity, direction):
        self.track_event('movement', None, {
            'x': x, 'y': y, 'velocity': velocity, 'direction': direction
        })
        
    def track_coin_collection(self, coin_value, total_coins, level):
        self.track_event('coin_collected', None, {
            'coin_value': coin_value, 'total_coins': total_coins, 'level': level
        })
        
    def track_enemy_interaction(self, enemy_type, interaction_type, success):
        self.track_event('enemy_interaction', None, {
            'enemy_type': enemy_type, 'interaction': interaction_type, 'success': success
        })
        
    def export_events(self, filename):
        df = pd.DataFrame(self.events)
        df.to_csv(f'data/{filename}', index=False)
        return len(self.events)
