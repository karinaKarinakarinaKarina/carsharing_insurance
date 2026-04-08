# insurance/tests/test_integration.py
import pytest
import json
from django.urls import reverse

pytestmark = pytest.mark.django_db

class TestIntegration:
    def test_full_prediction_flow(self, client, sample_trip_data):
        response = client.post(
            reverse('predict'),
            data=json.dumps(sample_trip_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.json()
        assert 'risk_probability' in data
        assert 'insurance_price' in data

    def test_health_check_integration(self, client):
        response = client.get(reverse('health'))
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'