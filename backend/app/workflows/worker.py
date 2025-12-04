"""Simple RQ worker launcher for SearchOne.

Usage (PowerShell):
  $env:REDIS_URL = "redis://localhost:2002"
  python worker.py

This will start an RQ worker listening on the `searchone` queue.
"""
import os
from redis import Redis
from rq import Worker, Queue
from rq.worker import SimpleWorker
try:
  from rq import connections as rq_connections
  Connection = rq_connections.Connection
except Exception:
  # older/newer rq versions may expose Connection differently; fallback to contextlib.nullcontext
  from contextlib import nullcontext as Connection

listen = ['searchone']

def run_worker():
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:2002')
    redis_conn = Redis.from_url(redis_url)
    with Connection(redis_conn):
        queues = [Queue(name, connection=redis_conn) for name in listen]
        # On Windows, fork is unavailable; SimpleWorker avoids forking.
        if os.name == "nt":
            worker = SimpleWorker(queues, connection=redis_conn)
        else:
            worker = Worker(queues, connection=redis_conn)
        worker.work()

if __name__ == '__main__':
    run_worker()
