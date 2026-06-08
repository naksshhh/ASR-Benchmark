import os
import pandas as pd
import numpy as np
import re
import sys
import glob

# Ensure stdout uses UTF-8 for console output and add project root to path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project normalization and language helper functions
from banking_asr_eval.metrics.normalize import normalize
from banking_asr_eval.metrics.codeswitching import detect_word_language
from banking_asr_eval.metrics.core import get_word_alignments

# Banking entities list for entity deletion classification
BANK_ENTITIES = {
    "hdfc", "sbi", "cibil", "aadhaar", "emi", "upi", "pin", "account", 
    "statement", "otp", "loan", "card", "bank", "pay", "payment", 
    "transfer", "balance", "rupee", "rupees", "co-win", "cowin", 
    "pan", "epfo", "uan", "m-pin", "mpin", "jio", "paypal", "amazon", 
    "paytm", "gpay", "phonepe"
}

# Number words for classification
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", 
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", 
    "eighty", "ninety", "hundred", "thousand", "lakh", "crore", "million", "billion",
    "एक", "दो", "तीन", "चार", "पाँच", "छः", "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह", 
    "तेरह", "चौदह", "पन्द्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस", "तीस", "चालीस", 
    "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे", "सौ", "हज़ार", "लाख", "करोड़"
}

def is_number_word(word):
    if re.search(r"\d", word):
        return True
    return word.lower() in NUMBER_WORDS

def categorize_alignment_chunk(chunk):
    op = chunk["type"]
    ref_words = chunk["ref_words"]
    hyp_words = chunk["hyp_words"]
    
    if op == "insert":
        return "hallucination"
        
    elif op == "delete":
        # Check if any ref word is a banking entity
        if any(w.lower() in BANK_ENTITIES for w in ref_words):
            return "entity_deletion"
        # Check if any ref word is numeric
        if any(is_number_word(w) for w in ref_words):
            return "number_substitution"
        # Check if ref word is in English (Latin script)
        if any(detect_word_language(w) == "english" for w in ref_words):
            return "codeswitched_miss"
        return "general_deletion"
        
    elif op == "substitute":
        # Check if any ref word is numeric or hyp word is numeric
        if any(is_number_word(w) for w in ref_words) or any(is_number_word(w) for w in hyp_words):
            return "number_substitution"
        # Check if any ref word is a banking entity
        if any(w.lower() in BANK_ENTITIES for w in ref_words):
            return "entity_deletion"
            
        # Script mismatch (e.g. English script reference, Devanagari hypothesis or vice-versa)
        ref_langs = [detect_word_language(w) for w in ref_words]
        hyp_langs = [detect_word_language(w) for w in hyp_words]
        
        if "english" in ref_langs and "hindi" in hyp_langs:
            return "hindi_romanized"
        if "hindi" in ref_langs and "english" in hyp_langs:
            return "hindi_romanized"
            
        # English term missed
        if all(detect_word_language(w) == "english" for w in ref_words):
            return "codeswitched_miss"
            
        return "general_substitution"
        
    return "equal"

def analyze_model_errors(model_name, df_model):
    print(f"\n=================== Analyzing {model_name} ===================")
    total_samples = len(df_model)
    print(f"Total samples evaluated: {total_samples}")
    
    categories = {
        "number_substitution": 0,
        "entity_deletion": 0,
        "codeswitched_miss": 0,
        "hindi_romanized": 0,
        "hallucination": 0,
        "general_deletion": 0,
        "general_substitution": 0
    }
    
    total_mismatch_words = 0
    
    for _, row in df_model.iterrows():
        ref = str(row.get("reference", ""))
        hyp = str(row.get("hypothesis", ""))
        
        if pd.isna(row.get("reference")) or not ref.strip():
            continue
            
        # Run alignment on normalized texts
        try:
            alignments = get_word_alignments(ref, hyp, pre_normalize=True)
            for chunk in alignments:
                cat = categorize_alignment_chunk(chunk)
                if cat != "equal":
                    # Weight by the number of words in mismatch (max of ref/hyp length)
                    weight = max(len(chunk["ref_words"]), len(chunk["hyp_words"]))
                    categories[cat] += weight
                    total_mismatch_words += weight
        except Exception as e:
            pass
            
    print(f"Total mismatch errors (in words): {total_mismatch_words}")
    if total_mismatch_words > 0:
        print("Error category breakdown:")
        for cat, val in categories.items():
            pct = val / total_mismatch_words * 100
            print(f"  - {cat:<22}: {val:>4} words ({pct:.2f}%)")
            
        # Specifically calculate how much script limits drive errors for IndicWav2Vec
        if "indicwav2vec" in model_name:
            script_limit_errors = categories["hindi_romanized"] + categories["codeswitched_miss"]
            pct_limit = script_limit_errors / total_mismatch_words * 100
            print(f"  [Insight] Script vocabulary limit (English words in Latin script) drives {pct_limit:.2f}% of all errors.")
    else:
        print("  No errors found (0% WER)")
        
    return categories

def main():
    # Load all synthetic_100 results files from results/ directory
    # We will identify synthetic results by sample count = 100 or 200, and dataset containing 'synthetic_100'
    files = glob.glob("results/eval_results_*.csv")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'dataset' in df.columns and any('synthetic_100' in str(d) for d in df['dataset'].unique()):
                dfs.append(df)
        except Exception:
            pass
            
    if not dfs:
        print("No synthetic_100 result files found. Make sure evaluations on synthetic_100.json have been run.")
        return
        
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Analyze each model
    models = df_all['model'].unique()
    results = {}
    for model in sorted(models):
        df_model = df_all[df_all['model'] == model]
        results[model] = analyze_model_errors(model, df_model)
        
    # Write report to text file
    report_path = "results/plots/error_analysis_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ASR ERROR ANALYSIS REPORT ON SYNTHETIC 100 HINGLISH\n")
        f.write("====================================================\n\n")
        for model, cats in sorted(results.items()):
            f.write(f"Model: {model}\n")
            f.write("-" * 40 + "\n")
            total = sum(cats.values())
            f.write(f"Total mismatch errors (words): {total}\n")
            if total > 0:
                for c, val in cats.items():
                    f.write(f"  - {c:<22}: {val:>4} words ({val/total*100:.2f}%)\n")
                if "indicwav2vec" in model:
                    script_limit = cats["hindi_romanized"] + cats["codeswitched_miss"]
                    f.write(f"  * Script vocabulary limit (Latin script mismatch) drives {script_limit/total*100:.2f}% of errors.\n")
            else:
                f.write("  No errors found.\n")
            f.write("\n")
            
    print(f"\nSaved detailed analysis report to: {report_path}")

if __name__ == "__main__":
    main()
