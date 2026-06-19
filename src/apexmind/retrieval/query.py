"""Build search queries from a question and its options."""
from __future__ import annotations

import re
from typing import Dict, List


def build_queries(question: str, options: Dict[str, str]) -> List[str]:
    """Return a small ordered list of queries, most targeted first.

    Heuristic: the bare question is the best Wikipedia query (it usually names the
    entity). We also add a question + shortest-option variant to disambiguate.
    """
    q = re.sub(r"\s+", " ", str(question)).strip().rstrip("?").strip()
    queries: List[str] = [q]

    # Add the shortest option as a hint (often the key entity / value).
    opt_texts = [o.strip() for o in options.values() if o and o.strip()]
    if opt_texts:
        shortest = min(opt_texts, key=len)
        if len(shortest) <= 60:
            queries.append(f"{q} {shortest}")

    # De-duplicate, preserve order.
    seen, out = set(), []
    for item in queries:
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out
