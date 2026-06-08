import pandas as pd
import numpy as np
import sys

# Reconfigure stdout to use UTF-8 for console printing
sys.stdout.reconfigure(encoding='utf-8')

csv_path = r"c:\Users\naksh\OneDrive\Desktop\Sem 6\Krim\ASR-Benchmark\results\eval_results_20260606_103532.csv"
df = pd.read_csv(csv_path)

out_lines = []
def print_log(s=""):
    print(s)
    out_lines.append(str(s))

print_log("Columns in CSV: " + str(list(df.columns)))
print_log("\nUnique models in CSV: " + str(list(df['model'].unique())))

print_log("\nMean WER by model:")
print_log(df.groupby('model')['wer'].mean())

print_log("\nMean CER by model:")
print_log(df.groupby('model')['cer'].mean())

whisper_models = [m for m in df['model'].unique() if 'whisper' in str(m)]
for model in whisper_models:
    print_log(f"\n=================== {model} ===================")
    m_df = df[df['model'] == model]
    print_log(f"Total samples: {len(m_df)}")
    high_wer = m_df[m_df['wer'] > 1.0]
    print_log(f"Samples with WER > 1.0: {len(high_wer)} ({len(high_wer)/len(m_df)*100:.2f}%)")
    
    # Let's print the top 10 rows with highest insertions
    print_log("\nTop 10 rows with highest insertions:")
    top_ins = m_df.sort_values(by='insertions', ascending=False).head(10)
    for idx, row in top_ins.iterrows():
        print_log(f"Audio ID: {row['audio_id']}")
        print_log(f"Ref: {row['reference']}")
        print_log(f"Hyp: {row['hypothesis']}")
        print_log(f"WER: {row['wer']:.2f} | Ins: {row['insertions']} | Del: {row['deletions']} | Sub: {row['substitutions']}")
        print_log("-" * 50)
        
    print_log("\nTop 10 rows with highest deletions:")
    top_del = m_df.sort_values(by='deletions', ascending=False).head(10)
    for idx, row in top_del.iterrows():
        print_log(f"Audio ID: {row['audio_id']}")
        print_log(f"Ref: {row['reference']}")
        print_log(f"Hyp: {row['hypothesis']}")
        print_log(f"WER: {row['wer']:.2f} | Ins: {row['insertions']} | Del: {row['deletions']} | Sub: {row['substitutions']}")
        print_log("-" * 50)

    # Let's also see some normal/low WER rows for whisper to see what they look like
    print_log("\nTop 5 rows with lowest WER (non-zero):")
    low_wer = m_df[m_df['wer'] > 0].sort_values(by='wer', ascending=True).head(5)
    for idx, row in low_wer.iterrows():
        print_log(f"Audio ID: {row['audio_id']}")
        print_log(f"Ref: {row['reference']}")
        print_log(f"Hyp: {row['hypothesis']}")
        print_log(f"WER: {row['wer']:.2f}")
        print_log("-" * 50)

with open(r"c:\Users\naksh\OneDrive\Desktop\Sem 6\Krim\ASR-Benchmark\scratch\analysis_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

