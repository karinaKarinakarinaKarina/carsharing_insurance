from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.cache import cache
from django.utils import timezone
import json
import logging
from datetime import datetime

from .ml_service import predictor

logger = logging.getLogger(__name__)

# Хранилище статистики в памяти (для демо)
prediction_stats = {
    'total': 0,
    'risks': [],
    'prices': [],
    'latencies': []
}

@require_http_methods(["GET"])
def health_check(request):
    return JsonResponse({
        'status': 'ok',
        'timestamp': timezone.now().isoformat(),
        'model_loaded': predictor is not None
    })

@csrf_exempt
@require_http_methods(["POST"])
def predict_risk(request):
    global prediction_stats
    
    try:
        data = json.loads(request.body)
        
        cache_key = f"prediction_{hash(str(data))}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache HIT for key: {cache_key}")
            return JsonResponse(cached_result)
        
        price, risk, latency = predictor.calculate_price(data)
        
        # Сохраняем статистику
        prediction_stats['total'] += 1
        prediction_stats['risks'].append(risk)
        prediction_stats['prices'].append(price)
        prediction_stats['latencies'].append(latency)
        
        # Ограничиваем размер списка (последние 1000)
        if len(prediction_stats['risks']) > 1000:
            prediction_stats['risks'] = prediction_stats['risks'][-1000:]
            prediction_stats['prices'] = prediction_stats['prices'][-1000:]
            prediction_stats['latencies'] = prediction_stats['latencies'][-1000:]
        
        result = {
            'risk_probability': float(risk),
            'insurance_price': float(price),
            'latency_ms': latency,
            'timestamp': timezone.now().isoformat()
        }
        
        cache.set(cache_key, result, timeout=300)
        
        return JsonResponse(result)
    
    except json.JSONDecodeError as e:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["POST"])
def predict_batch(request):
    try:
        data = json.loads(request.body)
        requests_data = data.get('requests', [])
        
        results = []
        for req in requests_data:
            price, risk, latency = predictor.calculate_price(req)
            results.append({
                'risk_probability': float(risk),
                'insurance_price': float(price)
            })
        
        return JsonResponse({
            'results': results,
            'count': len(results),
            'timestamp': timezone.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_stats(request):
    global prediction_stats
    
    if prediction_stats['total'] == 0:
        return JsonResponse({
            'total_predictions': 0,
            'avg_risk': 0,
            'min_risk': 0,
            'max_risk': 0,
            'avg_price': 0,
            'min_price': 0,
            'max_price': 0,
            'avg_latency_ms': 0,
            'message': 'No predictions yet. Make some predictions first!'
        })
    
    return JsonResponse({
        'total_predictions': prediction_stats['total'],
        'avg_risk': round(sum(prediction_stats['risks']) / len(prediction_stats['risks']), 4),
        'min_risk': round(min(prediction_stats['risks']), 4),
        'max_risk': round(max(prediction_stats['risks']), 4),
        'avg_price': round(sum(prediction_stats['prices']) / len(prediction_stats['prices']), 2),
        'min_price': round(min(prediction_stats['prices']), 2),
        'max_price': round(max(prediction_stats['prices']), 2),
        'avg_latency_ms': round(sum(prediction_stats['latencies']) / len(prediction_stats['latencies']), 2),
        'timestamp': timezone.now().isoformat()
    })