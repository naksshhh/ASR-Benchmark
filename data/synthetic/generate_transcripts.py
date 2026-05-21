import json
import os
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

def generate_transcripts():
    try:
        # Initialize client. Requires GEMINI_API_KEY environment variable
        client = genai.Client()
    except Exception as e:
        print("Could not initialize Google GenAI client. Make sure GEMINI_API_KEY is set.")
        return

    transcripts = []
    for domain in BANKING_DOMAINS:
        for _ in range(15):  # 15 conversations per domain = 120 total
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
                for turn in convs:
                    transcripts.append({
                        "text": turn["text"],
                        "speaker": turn["speaker"],
                        "domain": domain
                    })
            except Exception as e:
                print(f"Error generating for domain {domain}: {e}")
                continue

    os.makedirs("data/synthetic", exist_ok=True)
    with open("data/synthetic/transcripts.json", "w") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(transcripts)} transcripts.")

if __name__ == "__main__":
    generate_transcripts()
