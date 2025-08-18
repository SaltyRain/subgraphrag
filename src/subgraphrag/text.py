# ---- simple embeddings and cosine ----
import re
import math
from collections import Counter

_RE = re.compile(r"\w+", flags=re.UNICODE)

def _tokenize(t: str) -> list[str]:
    return _RE.findall(t.lower())

def _bow(text: str) -> Counter:
    return Counter(_tokenize(text))

def _cosine_bow(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    inter = set(a) & set(b)
    num = sum(a[t]*b[t] for t in inter)
    da = math.sqrt(sum(v*v for v in a.values()))
    db = math.sqrt(sum(v*v for v in b.values()))
    return num / (da*db) if da and db else 0.0