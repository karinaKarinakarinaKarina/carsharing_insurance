# insurance/admin.py
from django.contrib import admin
from .models import PredictionLog, ModelMetrics

@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'user', 'predicted_risk', 'predicted_price', 'latency_ms']
    list_filter = ['timestamp']
    search_fields = ['user__username']
    readonly_fields = ['input_data', 'predicted_risk', 'predicted_price', 'latency_ms']

@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'model_version', 'auc_score', 'log_loss', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['model_name']