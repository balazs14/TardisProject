"""
api_cache.py
Daily cache for HTTP GET JSON responses from the Tardis API.

Cache layout:
    datasets/api_cache/YYYY-MM-DD/<url-path-derived-name>.json

One subdirectory per calendar day.  Each file stores the raw JSON payload
for one (url, params) combination.  Re-running on the same calendar day
returns the cached payload without hitting the network.

Public API:
    cached_get(url, params=None, headers=None, timeout=30) -> object
"""

import json
import logging
import pathlib
from datetime import date
from urllib.parse import urlencode, urlparse

import requests


logger = logging.getLogger(__name__)

_API_CACHE_ROOT = pathlib.Path(__file__).parent.parent / "datasets" / "api_cache"


def _cache_file(url: str, params: dict | None, cache_date: date) -> pathlib.Path:
    # Derive a readable filename from the URL path.
    # Strip the /v1/ prefix, replace '/' with '--'.
    # Example: https://api.tardis.dev/v1/exchanges/okex-options -> exchanges--okex-options.json
    path = urlparse(url).path.lstrip("/")
    for prefix in ("v1/", "v1"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    name = path.replace("/", "--") or "root"
    if params:
        param_str = urlencode(sorted(params.items()))
        name = f"{name}--{param_str}"
    return _API_CACHE_ROOT / cache_date.isoformat() / f"{name}.json"


def cached_get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
) -> object:
    """GET url and return parsed JSON, using a per-calendar-day on-disk cache.

    Cache hit: reads the .json file written on a previous call today.
    Cache miss: fetches from the API, writes the response to disk, returns it.
    """
    cache_file = _cache_file(url, params, date.today())
    if cache_file.exists():
        logger.debug("API cache hit: %s -> %s", url, cache_file)
        return json.loads(cache_file.read_text(encoding="utf-8"))

    logger.debug("API cache miss, fetching: %s params=%s", url, params)
    with requests.get(url, headers=headers, params=params, timeout=timeout) as resp:
        resp.raise_for_status()
        payload = resp.json()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    logger.debug("API cache written: %s", cache_file)
    return payload
