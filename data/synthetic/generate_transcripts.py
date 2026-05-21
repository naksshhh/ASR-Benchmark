import json
import os
import concurrent.futures
from google import genai

BANKING_DOMAINS = [
    "debt collection reminder — customer has overdue EMI",
    "loan application status inquiry — personal loan pending",
    "credit card limit increase request",
    "KYC update — Aadhaar linking",
    "NACH mandate setup for auto-debit",
    "loan foreclosure inquiry — prepayment charges",
    "account balance and transaction inquiry",
    "fixed deposit maturity reminder",
]

PROMPT_TEMPLATE = """Generate a realistic Indian banking call center conversation.

Domain: {domain}
Language: Hinglish (natural Hindi-English code-switching, like real Indian call centers)
Turns: 4-6 utterances (alternating agent/customer)

Requirements:
- Include specific numbers: account numbers (10-12 digits spoken aloud), 
  EMI amounts (₹5,000-₹50,000), dates, loan IDs
- Natural code-switching: "aapka outstanding balance hai fifteen thousand rupees"
- Include hesitations, self-corrections: "matlab... haan, toh aapka..."
- Customer names: common Indian names (Rahul Sharma, Priya Singh, etc.)
- Banking terms: EMI, NACH, CIBIL score, KYC, foreclosure, prepayment

Output: JSON array of utterances
[
  {{"speaker": "agent", "text": "..."}},
  {{"speaker": "customer", "text": "..."}}
]
Only JSON, no other text."""

def generate_for_domain(client, domain):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=PROMPT_TEMPLATE.format(domain=domain)
        )
        
        # Gemini often wraps JSON in markdown blocks, let's clean it up
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        convs = json.loads(text.strip())
        results = []
        for turn in convs:
            results.append({
                "text": turn["text"],
                "speaker": turn["speaker"],
                "domain": domain
            })
        return results
    except Exception as e:
        print(f"Error generating for domain {domain}: {e}")
        return []

def generate_transcripts():
    try:
        # Initialize client. Requires GEMINI_API_KEY environment variable
        client = genai.Client()
    except Exception as e:
        print("Could not initialize Google GenAI client. Make sure GEMINI_API_KEY is set.")
        return

    # Create a list of all 120 tasks (8 domains * 15 conversations)
    tasks = []
    for domain in BANKING_DOMAINS:
        for _ in range(15):
            tasks.append(domain)

    transcripts = []
    print(f"Starting {len(tasks)} generation tasks...")
    
    # Use ThreadPoolExecutor to parallelize API requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_domain = {executor.submit(generate_for_domain, client, domain): domain for domain in tasks}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_domain):
            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{len(tasks)} tasks completed")
                
            results = future.result()
            if results:
                transcripts.extend(results)

    os.makedirs("data/synthetic", exist_ok=True)
    with open("data/synthetic/transcripts.json", "w") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    print(f"Finished! Generated {len(transcripts)} transcripts.")

if __name__ == "__main__":
    generate_transcripts()
