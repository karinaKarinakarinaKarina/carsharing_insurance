# insurance/tests/test_ml_service.py
from functools import cache
import json

from django.urls import reverse

import pytest
import pandas as pd
from insurance.ml_service import InsuranceRiskPredictor

class TestMLService:
    def test_predictor_initialization(self):
        predictor = InsuranceRiskPredictor()
        assert predictor is not None
        assert predictor.model is not None
        assert predictor.preprocessor is not None

    def test_prepare_features_dict(self, sample_trip_data):
        predictor = InsuranceRiskPredictor()
        df = predictor.prepare_features(sample_trip_data)
        assert df is not None
        assert len(df) == 1
        assert 'trip_distance_km' in df.columns

    def test_prepare_features_missing_columns(self):
        predictor = InsuranceRiskPredictor()
        data = {'trip_distance_km': 10.0}
        df = predictor.prepare_features(data)
        assert df is not None
        assert df['trip_distance_km'].iloc[0] == 10.0
        assert df['trip_duration_min'].iloc[0] == 0

    def test_predict_risk(self, sample_trip_data):
        predictor = InsuranceRiskPredictor()
        risk, latency = predictor.predict_risk(sample_trip_data)
        assert 0 <= risk <= 1
        assert latency > 0

    def test_calculate_price(self, sample_trip_data):
        predictor = InsuranceRiskPredictor()
        price, original_risk, latency = predictor.calculate_price(sample_trip_data)
        assert price > 0
        assert 0 <= original_risk <= 1

    def test_django_ml_service_integration(self, client, sample_trip_data):
        """Проверка интеграции Django и ML сервиса"""
        response = client.post(
            reverse('predict'),
            data=json.dumps(sample_trip_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'risk_probability' in data
        assert 'insurance_price' in data
        assert data['risk_probability'] > 0
