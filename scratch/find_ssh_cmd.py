import os
import json

brain_dir = r"C:\Users\naksh\.gemini\antigravity-ide\brain"
keywords = ["ssh"]

found = 0
for root, dirs, files in os.walk(brain_dir):
    if "transcript.jsonl" in files:
        path = os.path.join(root, "transcript.jsonl")
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        content = str(data.get("content", ""))
                        tool_calls = str(data.get("tool_calls", ""))
                        text = content + " " + tool_calls
                        if "ssh" in text.lower():
                            # Print any tool_calls where a command containing 'ssh' was executed
                            if "ssh" in tool_calls.lower() or "ssh" in content.lower():
                                print(f"Path: {path}\nStep {data.get('step_index')}: {text[:300]}...\n")
                                found += 1
                                if found > 50:
                                    break
                    except Exception:
                        pass
        except Exception:
            pass
