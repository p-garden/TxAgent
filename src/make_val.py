# make_submission_val.py (validated A/B/C/D only)
import json, csv
from tqdm import tqdm
from agent_tooluniverse_gpt import run_agent

VAL_JSONL = "data/curebench_valset_pharse1.jsonl"
OUT_CSV  = "submission_val.csv"

VALID_CHOICES = {"A", "B", "C", "D"}

correct_count = 0
total_count = 0

with open(VAL_JSONL, "r", encoding="utf-8") as f, open(OUT_CSV, "w", newline="", encoding="utf-8") as w:
    wr = csv.writer(w)
    wr.writerow(["id", "prediction", "reasoning_trace", "choice", "correct_answer"])
    for line in tqdm(f, desc="infer(val)"):
        ex = json.loads(line)
        qid = ex["id"]
        qtxt = ex["question"]
        opts = ex.get("options", {})
        correct_answer = ex.get("correct_answer", "").strip().upper()

        res = run_agent(qtxt, opts, max_rounds=3, model="gpt-4o-mini")
        # Extract and validate final choice
        raw = (res.get("final_choice") or "").strip()[:1].upper()
        choice = raw if raw in VALID_CHOICES else "D"

        rationale = res.get("rationale", "")

        wr.writerow([
            qid,
            choice,        # prediction = 최종 선택지 (validated)
            rationale,     # reasoning_trace
            choice,        # choice = 최종 선택지 (validated)
            correct_answer
        ])

        total_count += 1
        if choice == correct_answer:
            correct_count += 1

accuracy = correct_count / total_count if total_count > 0 else 0
print(f"Accuracy: {accuracy:.4f}")
print("saved ->", OUT_CSV)