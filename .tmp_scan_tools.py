import json, os, re
from collections import Counter

bash_counter = Counter()
mcp_counter = Counter()

with open("E:/antenna/.tmp_jsonl_list.txt") as f:
    paths = [l.strip() for l in f if l.strip()]

# Convert MSYS paths /c/Users/... to C:/Users/...
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
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") != "tool_use":
                        continue
                    name = c.get("name", "")
                    inp = c.get("input") or {}
                    if name == "Bash":
                        cmd = (inp.get("command") or "").strip()
                        if not cmd:
                            continue
                        while True:
                            m = re.match(r"^[A-Z_][A-Z0-9_]*=\S+\s+", cmd)
                            if not m:
                                break
                            cmd = cmd[m.end():]
                        while True:
                            m = re.match(r"^(sudo|timeout\s+\d+\S*|nohup|exec)\s+", cmd)
                            if not m:
                                break
                            cmd = cmd[m.end():]
                        cmd = re.split(r"\s*(?:\|\||&&|\||;|>>?|<<?)\s*", cmd, maxsplit=1)[0].strip()
                        toks = cmd.split()
                        if not toks:
                            continue
                        first = os.path.basename(toks[0])
                        if len(toks) >= 2 and not toks[1].startswith("-") and re.match(r"^[a-zA-Z][a-zA-Z0-9_:.\-]*$", toks[1]):
                            key = f"{first} {toks[1]}"
                        else:
                            key = first
                        bash_counter[key] += 1
                    elif name.startswith("mcp__"):
                        mcp_counter[name] += 1
    except Exception as e:
        continue

print("=== BASH (top 60) ===")
for k, v in bash_counter.most_common(60):
    print(f"{v:5d}  {k}")
print()
print("=== MCP (top 30) ===")
for k, v in mcp_counter.most_common(30):
    print(f"{v:5d}  {k}")
