from diskcache import Cache
from pathlib import Path


CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache = Cache(directory=str(CACHE_DIR))


def cached_get(key: str):
    return cache.get(key)


def cached_set(key: str, value, ttl: int = None):
    cache.set(key, value, expire=ttl)