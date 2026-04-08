# insurance/models.py
from datetime import timedelta, timezone

from django.db import models
from django.contrib.auth.models import User

class PredictionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    input_data = models.JSONField()
    predicted_risk = models.FloatField()
    predicted_price = models.FloatField()
    latency_ms = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['user_id']),
        ]
    
    def __str__(self):
        return f"Prediction at {self.timestamp} - Risk: {self.predicted_risk:.4f}"

class ModelMetrics(models.Model):
    model_name = models.CharField(max_length=100)
    model_version = models.CharField(max_length=50)
    auc_score = models.FloatField()
    log_loss = models.FloatField()
    brier_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.model_name} v{self.model_version} - AUC: {self.auc_score:.4f}"
    