import re

blog_path = r"C:\Users\naksh\.gemini\antigravity-ide\brain\a64d54b3-330f-483b-93e8-f87109883e60\.system_generated\steps\2389\content.md"
with open(blog_path, "r", encoding="utf-8") as f:
    html = f.read()

# remove HTML tags to get raw text
raw_text = re.sub(r'<[^>]*>', ' ', html)
# replace multiple spaces/newlines
raw_text = re.sub(r'\s+', ' ', raw_text)

# Let's search for sentences containing keyword
keywords = ["clone", "install", "nemo", "requirement", "version", "pip", "branch", "prerequisite"]
sentences = re.split(r'\.|\?|\!', raw_text)
print(f"Found {len(sentences)} sentences.")
count = 0
for s in sentences:
    s_clean = s.strip()
    if any(k in s_clean.lower() for k in keywords):
        # check if it contains interesting info
        if len(s_clean) > 30 and len(s_clean) < 500:
            print(f"- {s_clean}")
            count += 1
            if count > 40:
                break
