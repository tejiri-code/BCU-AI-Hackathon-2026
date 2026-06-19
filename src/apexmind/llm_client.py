"""Thin client for a llama.cpp server's OpenAI-compatible API (chat + rerank)."""
from __future__ import annotations

from typing import Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_fixed


class LlamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8081, timeout: int = 180):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base}/health", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        top_p: float = 1.0,
    ) -> str:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed
        r = requests.post(f"{self.base}/v1/chat/completions", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[Dict]:
        """Call the llama.cpp /rerank endpoint. Returns [{index, relevance_score}]."""
        payload = {"query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        r = requests.post(f"{self.base}/rerank", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("results", [])
