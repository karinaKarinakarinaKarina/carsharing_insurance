import time
import logging
from django.utils.deprecation import MiddlewareMixin
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])

class MetricsMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request.start_time = time.time()
    
    def process_response(self, request, response):
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.path,
                status=response.status_code
            ).inc()
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=request.path
            ).observe(duration)
            
            logger.info(f"{request.method} {request.path} - Status: {response.status_code} - Duration: {duration:.3f}s")
        
        return response

class LoggingMiddleware(MiddlewareMixin):
    def process_request(self, request):
        logger.info(f"Request started: {request.method} {request.path} from {request.META.get('REMOTE_ADDR')}")
    
    def process_response(self, request, response):
        logger.info(f"Request completed: {request.method} {request.path} - {response.status_code}")
        return response