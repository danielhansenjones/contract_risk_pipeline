import logging

import redis

from shared.settings import settings

logger = logging.getLogger(__name__)


class JobQueue:

    def __init__(self):
        self._client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
        )
        self._key = settings.job_queue_key

    def enqueue(self, job_id: str) -> None:
        try:
            self._client.lpush(self._key, job_id)
        except redis.ConnectionError as exc:
            logger.error("queue: failed to enqueue job %s: %s", job_id, exc)
            raise

    def dequeue(self, timeout: int = 5) -> str | None:
        try:
            result = self._client.brpop(self._key, timeout=timeout)
        except redis.ConnectionError as exc:
            logger.error("queue: failed to dequeue: %s", exc)
            raise
        if result is None:
            return None
        _, job_id = result
        return job_id.decode() if isinstance(job_id, bytes) else job_id

    def depth(self) -> int:
        return self._client.llen(self._key)
