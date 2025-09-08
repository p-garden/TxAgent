# test_single.py
from agent_tooluniverse_gpt import run_agent

q = "A patient is taking venlafaxine. Which OTC supplement should be avoided due to risk of serotonin syndrome?"
opts = {"A":"Vitamin D", "B":"St. John's Wort", "C":"Omega-3", "D":"Calcium"}

res = run_agent(q, opts, max_rounds=3, model="gpt-4o-mini")
print("FINAL:", res["final_choice"])
print("RATIONALE:\n", res["rationale"])
print("TOOLS:")
for t in res["tools"]:
    print(" -", t["tool"], t["args"], "->", str(t["result"])[:120], "...")