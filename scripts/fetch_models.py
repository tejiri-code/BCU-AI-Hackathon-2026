"""Download additional <=8B models (GGUF) into models/. Picks Q4_K_M for the LLM
and a high-quality quant for the embedding/reranker."""
import os, sys
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import list_repo_files, hf_hub_download

DST = os.path.join(os.path.dirname(__file__), "..", "models")
DST = os.path.abspath(DST)
os.makedirs(DST, exist_ok=True)

# repo -> preferred quant substrings in priority order
JOBS = [
    ("Qwen/Qwen3-8B-GGUF",            ["q4_k_m"]),
    ("Qwen/Qwen3-Embedding-0.6B-GGUF",["f16", "q8_0", "bf16"]),
    ("gpustack/bge-reranker-v2-m3-GGUF",["f16", "q8_0", "q4_k_m"]),
]

def pick(files, prefs):
    g = [f for f in files if f.lower().endswith(".gguf")]
    # skip sharded/mmproj for these text/embed models
    g = [f for f in g if "mmproj" not in f.lower()]
    for p in prefs:
        for f in sorted(g):
            if p in f.lower():
                return f
    return sorted(g)[0] if g else None

for repo, prefs in JOBS:
    try:
        files = list_repo_files(repo)
    except Exception as e:
        print(f"LIST-ERR {repo}: {e}"); continue
    target = pick(files, prefs)
    if not target:
        print(f"NO-GGUF {repo}"); continue
    print(f"DOWNLOAD {repo} :: {target}", flush=True)
    path = hf_hub_download(repo_id=repo, filename=target, local_dir=DST)
    print(f"  -> {path}", flush=True)

print("DONE")
