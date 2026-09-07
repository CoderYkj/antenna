import json, os, re
from collections import Counter

ex_counter = Counter()

with open("E:/antenna/.tmp_jsonl_list.txt") as f:
    paths = [l.strip() for l in f if l.strip()]

def to_win(p):
    m = re.match(r"^/([a-zA-Z])/(.*)$", p)
    if m:
        return f"{m.group(1).upper()}:/{m.group(2)}"
    return p

for p in paths:
    p = to_win(p)
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message") or {}
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content") or []
                if not isinstance(content, list):
                    continue
                for c in content:
                    if not isinstance(c, dict) or c.get("type") != "tool_use":
                        continue
                    name = c.get("name", "")
                    if name != "Bash":
                        continue
                    cmd = ((c.get("input") or {}).get("command") or "").strip()
                    if not cmd:
                        continue
                    # Only look at non-trivial cmds: starting with python/pip/pm2 (interesting ones)
                    if re.match(r"^(python|pip|pm2)\b", cmd):
                        # Take first 2 tokens (e.g. "python cli.py" or "pip show numpy")
                        toks = cmd.split()
                        if len(toks) >= 2:
                            ex_counter[" ".join(toks[:2])] += 1
                        else:
                            ex_counter[toks[0]] += 1
    except Exception:
        continue

print("=== Exact 2-token forms for python/pip/pm2 (top 40) ===")
for k, v in ex_counter.most_common(40):
    print(f"{v:5d}  {k}")
