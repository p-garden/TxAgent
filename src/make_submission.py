# make_submission.py
import json, csv
from tqdm import tqdm
from agent_tooluniverse_gpt import run_agent

IN_JSONL = "data/curebench_testset_phase1.jsonl"  # 입력 경로
OUT_CSV  = "submission.csv"

with open(IN_JSONL, "r", encoding="utf-8") as f, open(OUT_CSV, "w", newline="", encoding="utf-8") as w:
    wr = csv.writer(w)
    wr.writerow(["id", "prediction", "choice", "reasoning"])  # 새로운 포맷

    for line in tqdm(f, desc="infer"):
        ex = json.loads(line)
        qid = ex["id"]
        qtxt = ex["question"]
        opts = ex.get("options", {})

        res = run_agent(qtxt, opts, max_rounds=3, model="gpt-4o-mini")

        # prediction: 모델이 생성한 원문 (전체)
        prediction = res.get("rationale", "")

        # choice: 최종 A/B/C/D 한 글자 (없으면 D로 강제)
        choice = (res.get("final_choice") or "D")[:1]
        if choice not in ["A", "B", "C", "D"]:
            choice = "D"

        # reasoning: reasoning_trace만 별도 저장 (여기서는 rationale과 동일하게 사용)
        reasoning = prediction

        wr.writerow([qid, prediction, choice, reasoning])

print("saved ->", OUT_CSV)