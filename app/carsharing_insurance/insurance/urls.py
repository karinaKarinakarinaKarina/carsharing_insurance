# insurance/urls.py
from django.urls import path
from . import views
from . import metrics

urlpatterns = [
    path('health/', views.health_check, name='health'),
    path('predict/', views.predict_risk, name='predict'),
    path('predict/batch/', views.predict_batch, name='predict_batch'),
    path('stats/', views.get_stats, name='stats'),
    path('metrics/', metrics.metrics_view, name='metrics'),
]