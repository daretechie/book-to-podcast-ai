workers = 1  # Reduce number of workers for free tier
worker_class = 'sync'
timeout = 120  # Increase timeout for long-running tasks
max_requests = 1000
max_requests_jitter = 50
preload_app = True
