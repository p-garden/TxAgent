# tools_index.py
import json
import importlib.resources as ir

DATA_FILES = [
    "opentarget_tools.json",
    "fda_drug_labeling_tools.json",
    "special_tools.json",
    "monarch_tools.json",
]

def load_all_tools():
    tools = []
    for fname in DATA_FILES:
        try:
            path = ir.files("tooluniverse.data") / fname
            with open(path, "r", encoding="utf-8") as f:
                part = json.load(f)
                # part는 list 또는 dict일 수 있음 → 통일해서 리스트로
                if isinstance(part, dict):
                    # dict 형태면 values()에 tool spec이 담겨있을 가능성
                    part = list(part.values())
                if isinstance(part, list):
                    tools.extend(part)
        except Exception as e:
            print(f"[warn] failed to read {fname}: {e}")
    return tools

def index_by_name(tools):
    idx = {}
    for t in tools:
        name = t.get("name") or t.get("tool_name") or t.get("id")
        if name:
            idx[name] = t
    return idx

if __name__ == "__main__":
    tools = load_all_tools()
    idx = index_by_name(tools)
    print("total tools:", len(idx))
    # 예시로 몇 개만 출력
    for i, (name, spec) in enumerate(idx.items()):
        print("-", name, ":", (spec.get("desc","") or spec.get("description","") or "")[:80])
        if i >= 9:
            break