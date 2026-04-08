# insurance/tests/test_models.py
import pytest
from insurance.models import PredictionLog, ModelMetrics

pytestmark = pytest.mark.django_db

class TestPredictionLog:
    def test_create_prediction_log(self):
        log = PredictionLog.objects.create(
            input_data={'test': 'data'},
            predicted_risk=0.05,
            predicted_price=125.0,
            latency_ms=8.5
        )
        assert log.id is not None
        assert log.predicted_risk == 0.05

class TestModelMetrics:
    def test_create_model_metrics(self):
        metrics = ModelMetrics.objects.create(
            model_name='LogisticRegression',
            model_version='v1.0.0',
            auc_score=0.7291,
            log_loss=0.1173,
            brier_score=0.0264
        )
        assert metrics.id is not None
        assert metrics.model_name == 'LogisticRegression'