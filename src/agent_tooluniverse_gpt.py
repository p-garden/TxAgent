# agent_tooluniverse_gpt.py
import json, re, os
from typing import Dict, Any, List
from openai import OpenAI
from tooluniverse import ToolUniverse
from tools_cache import cached_run
from dotenv import load_dotenv, find_dotenv

# Try to load .env explicitly
dotenv_path = find_dotenv(usecwd=True)  # 현재 작업 디렉토리 기준으로 탐색
if not dotenv_path:
    dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")

load_dotenv(dotenv_path)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(f"OPENAI_API_KEY not found. Tried path: {dotenv_path}")
load_dotenv()
client = OpenAI()
tu = ToolUniverse()

# Expanded FDA tool set
ALLOW_TOOLS = {
    # 기존 커스텀 툴
    "fda_label_lookup",
    "drug_interaction_check",
    "renal_dose_adjust",
    "hepatic_dose_adjust",
    "pregnancy_lactation_check",

    # FDA 계열 툴
    "FDA_get_drug_interactions_by_drug_name",
    "FDA_get_contraindications_by_drug_name",
    "FDA_get_do_not_use_info_by_drug_name",
    "FDA_get_pregnancy_effects_info_by_drug_name",
    "FDA_get_pregnancy_or_breastfeeding_info_by_drug_name",
    "FDA_get_risk_info_by_drug_name",
}

SYS = (
    "You are TxAgent-like therapeutics assistant.\n"
    "- Prioritize safety: contraindications, interactions, renal/hepatic dose adjustments, pregnancy/lactation.\n"
    "- If you need external data, issue a tool call using EXACTLY this format:\n"
    "  CALL <tool_name> <json-args>\n"
    "  (no extra text on that line)\n"
    "  Preferred tools: FDA_get_drug_interactions_by_drug_name, FDA_get_contraindications_by_drug_name, FDA_get_pregnancy_effects_info_by_drug_name.\n"
    "- Otherwise provide a brief justification (no step-by-step chain-of-thought) and a final multiple-choice letter.\n"
    "- You MUST perform at least one tool call before giving the final answer.\n"

)

def _ask(messages, model="gpt-4o-mini", temperature=0.2):
    return client.chat.completions.create(
        model=model, temperature=temperature, messages=messages
    ).choices[0].message.content or ""

def _extract_call(text: str):
    m = re.match(r"\s*CALL\s+([A-Za-z0-9_\-]+)\s+(.*)$", text.strip(), flags=re.S)
    if not m:
        return None, None
    name = m.group(1)
    try:
        args = json.loads(m.group(2))
    except Exception:
        return name, None
    return name, args

def canonicalize_tool_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    low = name.strip().lower().replace(" ", "_")
    # Common aliases → FDA label tools
    alias_map = {
        "drug_interaction_check": "FDA_get_drug_interactions_by_drug_name",
        "drug_interactions": "FDA_get_drug_interactions_by_drug_name",
        "druginteraction": "FDA_get_drug_interactions_by_drug_name",
        "interaction_check": "FDA_get_drug_interactions_by_drug_name",
        "ddi_check": "FDA_get_drug_interactions_by_drug_name",
        "contraindications_check": "FDA_get_contraindications_by_drug_name",
        "pregnancy_check": "FDA_get_pregnancy_effects_info_by_drug_name",
        "pregnancy_lactation_check": "FDA_get_pregnancy_or_breastfeeding_info_by_drug_name",
        "risk_info": "FDA_get_risk_info_by_drug_name",
    }
    return alias_map.get(low, name)

def adapt_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    # Normalize different arg styles to each tool's schema
    def pick_drug(a):
        if not isinstance(a, dict):
            return {"drug_name": str(a)}
        if "drug_name" in a and isinstance(a["drug_name"], str):
            out = {"drug_name": a["drug_name"]}
            for k in ("limit", "skip"):
                if k in a:
                    out[k] = a[k]
            return out
        for alias in ("drug", "name", "query"):
            if alias in a and isinstance(a[alias], str):
                return {"drug_name": a[alias]}
        if "medications" in a and isinstance(a["medications"], list) and a["medications"]:
            return {"drug_name": str(a["medications"][0])}
        if "drug1" in a:
            return {"drug_name": str(a["drug1"])}
        if a:
            return {"drug_name": str(next(iter(a.values())))}
        return {"drug_name": ""}

    if name in {
        "FDA_get_drug_interactions_by_drug_name",
        "FDA_get_contraindications_by_drug_name",
        "FDA_get_do_not_use_info_by_drug_name",
        "FDA_get_pregnancy_effects_info_by_drug_name",
        "FDA_get_pregnancy_or_breastfeeding_info_by_drug_name",
        "FDA_get_risk_info_by_drug_name",
    }:
        return pick_drug(args)

    return args

def run_agent(question: str, options: Dict[str, str], max_rounds=4, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"Question:\n{question}\n\nOptions:\n{json.dumps(options, ensure_ascii=False)}"}
    ]
    trace: List[Dict[str, Any]] = []

    for _ in range(max_rounds):
        out = _ask(messages, model=model)
        name, args = _extract_call(out)
        canon = canonicalize_tool_name(name) if name else None

        if canon:
            if canon not in ALLOW_TOOLS:
                result = {"error": f"tool '{canon}' not allowed", "allowed": sorted(list(ALLOW_TOOLS))}
            elif args is None:
                result = {"error": "invalid JSON args"}
            else:
                try:
                    fixed_args = adapt_args(canon, args)
                    result = cached_run(tu, canon, **fixed_args)
                    if result is None:
                        result = {"error": "tool returned null (no result)."}
                except Exception as e:
                    result = {"error": str(e)}

            trace.append({"tool": canon, "original_tool": name, "args": args, "result": result})
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": f"[{canon} RESULT]\n{json.dumps(result, ensure_ascii=False)}\nUse it."})
            # small nudge if tool failed
            if isinstance(result, dict) and result.get("error"):
                messages.append({"role": "user", "content": "Tool output unavailable. Proceed using clinical knowledge and choose the final letter."})
            continue

        # LLM이 최종 답/요약을 냈다고 판단
        messages.append({"role": "assistant", "content": out})

        # 마지막에 선택지 문자만 강제 추출
        m = re.findall(r"\b([A-Z])\b", out)
        if m:
            return {"final_choice": m[-1], "rationale": out, "tools": trace}

        # 한 글자 강제 요구
        messages.append({"role": "user", "content": "Based on the tool results above, output ONLY the final choice letter (A/B/C/D). No explanation, no reasoning, no additional text."})
        forced = _ask(messages, model=model, temperature=0).strip()[:1]
        return {"final_choice": forced or "", "rationale": out, "tools": trace}

    return {"final_choice": "D", "rationale": "max_rounds_reached", "tools": trace}