# insurance/ml_service.py
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

class InsuranceRiskPredictor:
    def __init__(self, model_path=None, preprocessor_path=None):
        if model_path is None:
            model_path = BASE_DIR / 'models' / 'best_model.pkl'
        if preprocessor_path is None:
            preprocessor_path = BASE_DIR / 'models' / 'preprocessor.pkl'
        
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        self.boost_factor = 30.0
        self.min_risk = 0.10
        self.max_risk = 0.50
    
    def prepare_features(self, data):
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Input must be dict or DataFrame")
        
        required_features = [
            'trip_distance_km', 'trip_duration_min', 'demand_coefficient', 'trip_cost_rub',
            'driver_age', 'driving_experience_years', 'max_payment_delay_days', 'num_prev_contracts',
            'vehicle_mileage_km', 'vehicle_manufacture_year', 'engine_horsepower',
            'avg_speed_kmh', 'avg_longitudinal_accel', 'avg_lateral_accel', 'avg_brake_pressure',
            'steering_variability', 'avg_throttle_pos', 'lane_deviation', 'phone_usage_probability',
            'avg_headway_distance', 'avg_reaction_time_sec', 'hard_brakes_count',
            'hard_accelerations_count', 'sharp_turns_count', 'hour', 'day_of_week_num',
            'is_weekend', 'day_of_month', 'experience_ratio', 'hard_events_per_min',
            'night_and_inexperienced', 'style_weather_risk'
        ]
        
        categorical_features = [
            'weather_condition', 'gender', 'marital_status', 'engine_type',
            'vehicle_category', 'manufacturing_country', 'driving_style',
            'time_of_day_group', 'age_group'
        ]
        
        for col in required_features:
            if col not in df.columns:
                df[col] = 0
        
        for col in categorical_features:
            if col not in df.columns:
                df[col] = 'unknown'
        
        return df
    
    def predict_risk(self, data):
        start_time = datetime.now()
        
        df = self.prepare_features(data)
        X = self.preprocessor.transform(df)
        risk_probability = self.model.predict_proba(X)[:, 1]
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        if isinstance(data, dict):
            return risk_probability[0], latency_ms
        return risk_probability, latency_ms
    
    def calculate_price(self, data, base_price=100, risk_coefficient=500):
        risk, latency = self.predict_risk(data)
        
        boosted_risk = risk * self.boost_factor
        boosted_risk = max(boosted_risk, self.min_risk)
        boosted_risk = min(boosted_risk, self.max_risk)
        
        insurance_price = base_price + boosted_risk * risk_coefficient
        
        return insurance_price, risk, latency


predictor = InsuranceRiskPredictor()