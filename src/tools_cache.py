# tools_cache.py
import json, os, hashlib
CACHE_FILE = "tool_cache.json"
_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        _cache = json.load(open(CACHE_FILE, "r", encoding="utf-8"))
    except Exception:
        _cache = {}

def _key(name, kwargs):
    s = name + "|" + json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode()).hexdigest()

def cached_run(tu, name, **kwargs):
    k = _key(name, kwargs)
    if k in _cache:
        return _cache[k]
    out = tu.run(name, **kwargs)
    _cache[k] = out
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False)
    return out