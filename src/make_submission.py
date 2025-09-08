# make_submission.py
import json, csv
from tqdm import tqdm
from agent_tooluniverse_gpt import run_agent

IN_JSONL = "data/curebench_testset_phase1.jsonl"  # 경로 맞춰 수정
OUT_CSV  = "submission.csv"

with open(IN_JSONL, "r", encoding="utf-8") as f, open(OUT_CSV, "w", newline="", encoding="utf-8") as w:
    wr = csv.writer(w)
    wr.writerow(["id", "prediction", "reasoning_trace", "choice"])   # 대회 포맷 가정 (페이지에서 컬럼명 확인)
    for line in tqdm(f, desc="infer"):
        ex = json.loads(line)
        qid = ex["id"]; qtxt = ex["question"]; opts = ex.get("options", {})
        res = run_agent(qtxt, opts, max_rounds=3, model="gpt-4o-mini")
        FINAL = (res.get("final_choice") or "D")[:1]
        if FINAL not in ["A", "B", "C", "D"]:
            FINAL = "D"
        RATIONALE = res.get("rationale", "")
        wr.writerow([qid, FINAL, RATIONALE, FINAL])

print("saved ->", OUT_CSV)