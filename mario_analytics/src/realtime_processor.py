import asyncio
import websockets
import json
from collections import defaultdict

class RealtimeAnalytics:
    def __init__(self):
        self.active_sessions = defaultdict(dict)
        self.metrics_cache = {}
        
    async def handle_game_event(self, websocket, path):
        async for message in websocket:
            try:
                data = json.loads(message)
                await self.process_event(data, websocket)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({'error': 'Invalid JSON'}))
                
    async def process_event(self, event, websocket):
        event_type = event.get('type')
        
        if event_type == 'player_death':
            await self.analyze_death_pattern(event, websocket)
        elif event_type == 'level_complete':
            await self.analyze_completion_metrics(event, websocket)
        elif event_type == 'powerup_collected':
            await self.track_powerup_effectiveness(event, websocket)
            
    async def analyze_death_pattern(self, event, websocket):
        # Real-time difficulty adjustment logic
        death_location = event.get('location', {})
        if self.is_death_hotspot(death_location):
            suggestion = {'type': 'difficulty_adjust', 'action': 'reduce', 'area': death_location}
            await websocket.send(json.dumps(suggestion))
            
    def is_death_hotspot(self, location):
        # Simplified hotspot detection
        return location.get('deaths_in_area', 0) > 3
