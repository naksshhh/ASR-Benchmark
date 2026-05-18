"""
Banking dialogue scripts for synthetic data generation.

These templates represent the kinds of utterances that occur in Indian
banking call centers. Used to generate test audio via TTS.
"""

import random
from typing import Dict, List, Tuple


# ─── Banking Scenario Templates ──────────────────────────────────────────────
# Each template is a (language, text) tuple. Language can be:
# "hindi", "english", "mixed" (Hindi-English code-switching)

ACCOUNT_INQUIRY: List[Tuple[str, str]] = [
    ("hindi", "मेरा अकाउंट नंबर है नौ दो पांच तीन एक चार सात आठ छह"),
    ("hindi", "मुझे अपने बचत खाते का बैलेंस जानना है"),
    ("mixed", "मेरा savings account balance कितना है"),
    ("mixed", "account number nine two five three one four seven eight six बता दीजिए"),
    ("english", "I want to check my savings account balance"),
    ("english", "My account number is nine two five three one four seven eight six"),
    ("mixed", "please मेरा account statement भेज दीजिए last three months का"),
    ("hindi", "मेरे खाते में पिछले महीने कितने पैसे आए थे"),
    ("mixed", "mera current account ka balance check karna hai"),
    ("mixed", "account number nau do paanch teen ek chaar saat aath chhe"),
]

LOAN_EMI: List[Tuple[str, str]] = [
    ("hindi", "मेरी EMI कितनी है इस महीने की"),
    ("mixed", "home loan EMI amount fifty thousand rupees है"),
    ("mixed", "mera loan amount hai fifty thousand rupees"),
    ("english", "My home loan EMI is due on the fifteenth of every month"),
    ("mixed", "personal loan का interest rate कितना है"),
    ("hindi", "मेरा लोन अकाउंट नंबर एल एच दो शून्य तीन पांच सात आठ"),
    ("mixed", "EMI due date पंद्रह तारीख है हर महीने"),
    ("english", "I want to prepay my car loan of two lakh rupees"),
    ("mixed", "loan outstanding amount teen lakh paanch hazaar rupees hai"),
    ("mixed", "mujhe education loan ke baare mein jaankari chahiye"),
]

CARD_ACTIVATION: List[Tuple[str, str]] = [
    ("mixed", "mera credit card activate karna hai"),
    ("hindi", "मेरा नया डेबिट कार्ड आया है उसे चालू करना है"),
    ("mixed", "card number four two seven one eight three five six nine zero है"),
    ("english", "I need to activate my new HDFC credit card"),
    ("mixed", "CVV number teen chaar paanch है"),
    ("mixed", "card expiry date December twenty twenty seven है"),
    ("hindi", "मुझे अपने कार्ड का पिन बदलना है"),
    ("mixed", "credit card limit badhaana hai fifty thousand se one lakh tak"),
    ("english", "My card was declined at the ATM please check"),
    ("mixed", "debit card ka PIN reset karna hai"),
]

KYC_VERIFICATION: List[Tuple[str, str]] = [
    ("mixed", "mera Aadhaar number hai barah sau thirty four fifty six seventy eight ninety"),
    ("hindi", "मेरा आधार नंबर बारह सौ चौंतीस छप्पन अठहत्तर नब्बे है"),
    ("mixed", "PAN card number ABCDE one two three four F है"),
    ("english", "I need to update my KYC documents"),
    ("mixed", "Aadhaar se linked mobile number change karna hai"),
    ("mixed", "mera CIBIL score kitna hai"),
    ("hindi", "मुझे अपना पता बदलना है बैंक के रिकॉर्ड में"),
    ("english", "My PAN number is Alpha Bravo Charlie Delta Echo one two three four Foxtrot"),
    ("mixed", "voter ID se KYC update ho sakta hai kya"),
    ("mixed", "address proof ke liye Aadhaar card chalega"),
]

FUND_TRANSFER: List[Tuple[str, str]] = [
    ("mixed", "mujhe paanch hazaar rupees transfer karne hain SBI account mein"),
    ("hindi", "NEFT से दस हज़ार रुपये भेजने हैं"),
    ("mixed", "IFSC code SBIN zero zero one two three four है"),
    ("english", "I want to transfer twenty five thousand rupees via RTGS"),
    ("mixed", "UPI se payment karna hai paanch sau rupees"),
    ("mixed", "beneficiary ka account number ek do teen chaar paanch chhe saat aath nau hai"),
    ("hindi", "IMPS से तुरंत पैसे भेजने हैं दो हज़ार रुपये"),
    ("mixed", "NEFT transfer कितने time में complete होता है"),
    ("english", "Please add this account as a beneficiary for NEFT transfers"),
    ("mixed", "UPI ID hai naksh at oksbi"),
]

COMPLAINT_REGISTRATION: List[Tuple[str, str]] = [
    ("mixed", "ATM se paise nahi nikle lekin account se kat gaye"),
    ("hindi", "मेरे खाते से गलत पैसे कट गए हैं"),
    ("mixed", "unauthorized transaction hua hai mere account mein"),
    ("english", "I want to report a fraudulent transaction on my credit card"),
    ("mixed", "ATM mein card fass gaya hai Kotak Mahindra bank ka"),
    ("hindi", "मेरा चेक बाउंस हो गया कारण बताइए"),
    ("mixed", "online fraud ho gaya hai paanch hazaar rupees ka"),
    ("english", "My debit card was used without my knowledge at a POS terminal"),
    ("mixed", "complaint number kya hai mera previous complaint ka"),
    ("mixed", "OTP nahi aa raha hai mobile pe transaction ke liye"),
]


ALL_SCENARIOS: Dict[str, List[Tuple[str, str]]] = {
    "account_inquiry": ACCOUNT_INQUIRY,
    "loan_emi": LOAN_EMI,
    "card_activation": CARD_ACTIVATION,
    "kyc_verification": KYC_VERIFICATION,
    "fund_transfer": FUND_TRANSFER,
    "complaint_registration": COMPLAINT_REGISTRATION,
}


def get_random_scripts(
    n: int = 100,
    scenarios: List[str] = None,
    languages: List[str] = None,
) -> List[Dict]:
    """
    Get n random banking scripts for synthetic data generation.

    Args:
        n: Number of scripts to generate
        scenarios: Filter by scenario type. None = all.
        languages: Filter by language. None = all.

    Returns:
        List of dicts with keys: scenario, language, text, script_id
    """
    pool = []
    for scenario_name, templates in ALL_SCENARIOS.items():
        if scenarios and scenario_name not in scenarios:
            continue
        for lang, text in templates:
            if languages and lang not in languages:
                continue
            pool.append({
                "scenario": scenario_name,
                "language": lang,
                "text": text,
            })

    if not pool:
        raise ValueError("No scripts match the given filters")

    scripts = []
    for i in range(n):
        script = random.choice(pool).copy()
        script["script_id"] = f"script_{i:04d}"
        scripts.append(script)

    return scripts
