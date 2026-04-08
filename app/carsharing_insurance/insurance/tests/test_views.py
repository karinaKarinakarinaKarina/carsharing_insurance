# insurance/tests/test_views.py
import pytest
import json
from django.urls import reverse
from insurance.models import PredictionLog

pytestmark = pytest.mark.django_db

class TestHealthCheck:
    def test_health_check(self, client):
        response = client.get(reverse('health'))
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'

class TestPredictAPI:
    def test_predict_success(self, client, sample_trip_data):
        response = client.post(
            reverse('predict'),
            data=json.dumps(sample_trip_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.json()
        assert 'risk_probability' in data
        assert 'insurance_price' in data

    def test_predict_low_risk(self, client, low_risk_trip_data):
        response = client.post(
            reverse('predict'),
            data=json.dumps(low_risk_trip_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.json()
        assert data['risk_probability'] < 0.05

    def test_predict_invalid_json(self, client):
        response = client.post(
            reverse('predict'),
            data='invalid json',
            content_type='application/json'
        )
        assert response.status_code == 400
        assert 'error' in response.json()

    def test_predict_empty_data(self, client):
        response = client.post(
            reverse('predict'),
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.json()
        assert data['insurance_price'] > 0