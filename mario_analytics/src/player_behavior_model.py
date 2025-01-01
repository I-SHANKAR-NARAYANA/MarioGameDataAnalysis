import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

class PlayerBehaviorAnalyzer:
    def __init__(self):
        self.churn_model = RandomForestClassifier(n_estimators=100)
        self.ltv_model = GradientBoostingRegressor(n_estimators=100)
        self.segment_model = KMeans(n_clusters=5)
        self.scaler = StandardScaler()
        
    def prepare_features(self, player_data):
        features = pd.DataFrame({
            'avg_session_length': player_data['session_lengths'].mean(),
            'total_playtime': player_data['session_lengths'].sum(),
            'completion_rate': player_data['levels_completed'] / player_data['levels_attempted'],
            'death_rate': player_data['total_deaths'] / player_data['total_playtime'],
            'coin_efficiency': player_data['coins_collected'] / player_data['total_playtime'],
            'powerup_usage': player_data['powerups_used'] / player_data['powerups_found'],
            'social_interactions': player_data['multiplayer_sessions'],
            'purchase_frequency': player_data['iap_count'],
            'days_since_last_session': player_data['days_since_last_session']
        }, index=[0])
        return features
        
    def predict_churn_probability(self, player_features):
        scaled_features = self.scaler.transform(player_features)
        return self.churn_model.predict_proba(scaled_features)[0][1]
        
    def predict_lifetime_value(self, player_features):
        scaled_features = self.scaler.transform(player_features)
        return self.ltv_model.predict(scaled_features)[0]
        
    def segment_player(self, player_features):
        scaled_features = self.scaler.transform(player_features)
        cluster = self.segment_model.predict(scaled_features)[0]
        
        segments = {
            0: "Casual Explorer", 1: "Hardcore Completionist", 
            2: "Social Gamer", 3: "Speedrunner", 4: "Whale Spender"
        }
        return segments.get(cluster, "Unknown")
        
    def train_models(self, training_data):
        X = self.scaler.fit_transform(training_data.drop(['churn', 'ltv'], axis=1))
        
        self.churn_model.fit(X, training_data['churn'])
        self.ltv_model.fit(X, training_data['ltv'])
        self.segment_model.fit(X)
        
        # Save models
        joblib.dump(self.churn_model, 'models/churn_model.pkl')
        joblib.dump(self.ltv_model, 'models/ltv_model.pkl')
        joblib.dump(self.segment_model, 'models/segment_model.pkl')
