import json

log_path = r"C:\Users\naksh\.gemini\antigravity-ide\brain\a64d54b3-330f-483b-93e8-f87109883e60\.system_generated\logs\transcript.jsonl"

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            content = str(data.get("content", ""))
            tool_calls = str(data.get("tool_calls", ""))
            text = content + " " + tool_calls
            if "ssh" in text.lower() and ("@" in text or "iitp" in text or "pem" in text):
                print(f"Index {data.get('step_index')}: {text[:500]}...\n")
        except Exception:
            pass
