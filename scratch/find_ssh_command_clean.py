import json
import re

log_path = r"C:\Users\naksh\.gemini\antigravity-ide\brain\a64d54b3-330f-483b-93e8-f87109883e60\.system_generated\logs\transcript.jsonl"
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            content = str(data.get("content", ""))
            tool_calls = str(data.get("tool_calls", ""))
            text = content + " " + tool_calls
            if "ssh" in text.lower():
                # find ssh commands with @ or pem
                matches = re.findall(r'ssh\s+[^\n]+', text, re.IGNORECASE)
                for m in matches:
                    print(f"Step {data.get('step_index')}: {m}")
        except Exception:
            pass
