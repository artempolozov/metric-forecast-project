M_1 = 'microservice-1'
M_2 = 'microservice-2'
M_3 = 'microservice-3'
M_4 = 'microservice-4'
M_5 = 'microservice-5'
M_6 = 'microservice-6'
M_7 = 'microservice-7'
M_8 = 'microservice-8'
M_9 = 'microservice-9'
M_10 = 'microservice-10'

CPU = 'CPU'
DISK = 'Disk'
MEMORY = 'Memory'
NETWORK = 'Network'

PRICE_ENDPOINT = '/price'
USER_ENDPOINT = '/users'
ORDER_ENDPOINT = '/orders'
INVENTORY_ENDPOINT = '/inventory'
PAYMENT_ENDPOINT = '/price'

GET_METHOD = 'GET'
PUT_METHOD = 'PUT'
POST_METHOD = 'POST'

INFRASTRUCTURE_METRICS = 'infrastructure_metrics'
INFRASTRUCTURE_METRICS_LOAD_PERCENT = 'infrastructure_metrics_load_percent'
INFRASTRUCTURE_METRICS_RESPONSE_TIME = 'infrastructure_metrics_response_time'
INFRASTRUCTURE_METRICS_RECOVERY_TIME = 'infrastructure_metrics_recovery'
HTTP_METRICS = 'http_metrics'
HTTP_METRICS_ERROR_RATE = 'http_metrics_error_rate'
HTTP_METRICS_AVG_RESPONSE_TIME = 'http_metrics_avg_response_time'
SERVICE_HEALTH_METRICS = 'service_health'

TARGETS = {
    INFRASTRUCTURE_METRICS_LOAD_PERCENT: 'load_percent',
    INFRASTRUCTURE_METRICS_RESPONSE_TIME: 'response_time_ms',
    INFRASTRUCTURE_METRICS_RECOVERY_TIME: 'recovery_time_ms',
    HTTP_METRICS_ERROR_RATE: 'error_rate',
    HTTP_METRICS_AVG_RESPONSE_TIME: 'avg_response_time_ms',
    SERVICE_HEALTH_METRICS: 'uptime_percent'
}

INFRASTRUCTURE_METRIC_NAMES = [CPU, DISK, MEMORY, NETWORK]
ENDPOINTS = [PRICE_ENDPOINT, USER_ENDPOINT, ORDER_ENDPOINT, INVENTORY_ENDPOINT, PAYMENT_ENDPOINT]
METHODS = [GET_METHOD, PUT_METHOD, POST_METHOD]
