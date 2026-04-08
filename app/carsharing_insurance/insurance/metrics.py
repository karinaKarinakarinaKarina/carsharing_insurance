from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from django.http import HttpResponse
from django.core.cache import cache
import psutil
import os

PREDICTION_COUNT = Counter('predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency', buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1])
ERROR_COUNT = Counter('errors_total', 'Total number of errors', ['error_type'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')
MODEL_LOADED = Gauge('model_loaded', 'Whether model is loaded')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')
SYSTEM_CPU = Gauge('system_cpu_percent', 'System CPU usage percent')
SYSTEM_MEMORY = Gauge('system_memory_percent', 'System memory usage percent')

def update_system_metrics():
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_MEMORY.set(psutil.virtual_memory().percent)

def metrics_view(request):
    update_system_metrics()
    return HttpResponse(generate_latest(), content_type=CONTENT_TYPE_LATEST)