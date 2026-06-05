import json

log_path = r"C:\Users\naksh\.gemini\antigravity-ide\brain\a64d54b3-330f-483b-93e8-f87109883e60\.system_generated\logs\transcript.jsonl"
keywords = ["ssh", "pem", "sbatch", "squeue", "param", "rudra"]

found = 0
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            content = str(data.get("content", ""))
            tool_calls = str(data.get("tool_calls", ""))
            text = content + " " + tool_calls
            if any(k in text.lower() for k in keywords):
                # Print a clean snippet of the matching line
                print(f"Index {data.get('step_index')}: {text[:300]}...\n")
                found += 1
                if found > 30:
                    break
        except Exception as e:
            pass
