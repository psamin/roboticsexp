"""API-equivalent cost of Claude Code transcripts (dedup by message id), as one costs.json item.
usage: cost.py <label> <jsonl>...   prints {"label", "usd", "tokens"}"""
import json
import sys

# $/M tokens: input, 5-min cache write (1.25x), 1-hour cache write (2x), cache read, output
PRICE = {"claude-opus-5-5": (4, 5, 8, 0.2, 20), "claude-sonnet-5-5": (2, 2.5, 4, 0.1, 10)}
usd, toks, seen = 0.0, 0, set()
for f in sys.argv[2:]:
    for line in open(f, errors="ignore"):
        m = json.loads(line).get("message")
        if not isinstance(m, dict) or "usage" not in m or m.get("model") == "<synthetic>" or m["id"] in seen:
            continue
        seen.add(m["id"])
        u, cc = m["usage"], m["usage"]["cache_creation"]
        n = (u["input_tokens"], cc["ephemeral_5m_input_tokens"], cc["ephemeral_1h_input_tokens"],
             u["cache_read_input_tokens"], u["output_tokens"])
        usd += sum(a * b for a, b in zip(n, PRICE[m["model"]])) / 1e6
        toks += sum(n)
print(json.dumps({"label": sys.argv[1], "usd": round(usd, 2), "tokens": toks}))
