import json, re, os
from typing import Dict, Any, List
from openai import OpenAI
from tooluniverse import ToolUniverse
from tools_cache import cached_run
from dotenv import load_dotenv, find_dotenv

# Try to load .env explicitly
dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")

load_dotenv(dotenv_path)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(f"OPENAI_API_KEY not found. Tried path: {dotenv_path}")
client = OpenAI()
tu = ToolUniverse()

# Expanded FDA tool set
ALLOW_TOOLS = {
    "fda_label_lookup",
    "drug_interaction_check",
    "renal_dose_adjust",
    "hepatic_dose_adjust",
    "pregnancy_lactation_check",  
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
    "- If a tool accepts a single `drug_name` but you must compare multiple options, call the tool MULTIPLE times (one per option) and clearly tie each call/result to an option letter.\n"
    "- Your reply must end with a line exactly of the form: `Final answer: X` where X is one of A/B/C/D.\n"
    "- Route tools by question type:\n"
    "  * interactions/병용 → FDA_get_drug_interactions_by_drug_name\n"
    "  * contraindications/금기 → FDA_get_contraindications_by_drug_name\n"
    "  * pregnancy/임신 → FDA_get_pregnancy_effects_info_by_drug_name\n"
    "  * breastfeeding/수유 → FDA_get_pregnancy_or_breastfeeding_info_by_drug_name\n"
    "  * warnings/주의/위험 → FDA_get_risk_info_by_drug_name\n"
    "- Otherwise provide a brief justification (no step-by-step chain-of-thought) and a final multiple-choice letter.\n"
)

def _ask(messages, model="gpt-4o-mini", temperature=0.2):
    """Sends a message sequence to the LLM and returns the response."""
    return client.chat.completions.create(
        model=model, temperature=temperature, messages=messages
    ).choices[0].message.content or ""

def _extract_call(text: str):
    """Extracts a tool call and its arguments from the LLM's response."""
    m = re.match(r"\s*CALL\s+([A-Za-z0-9_\-]+)\s+(.*)$", text.strip(), flags=re.S)
    if not m:
        return None, None
    name = m.group(1)
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        return name, None
    return name, args

def canonicalize_tool_name(name: str) -> str:
    """Normalizes tool names and handles aliases."""
    if not isinstance(name, str):
        return name
    low = name.strip().lower().replace(" ", "_")
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
    """
    다양한 인수 스타일을 표준화된 도구 스키마에 맞게 조정합니다.
    이 함수는 이제 중첩된 딕셔너리 및 여러 약물 이름과 같은
    더 넓은 범위의 입력 형식을 처리합니다.
    """
    if not args:
        return {"drug_name": ""}

    # 1. 간단한 딕셔너리가 아닌 인수를 처리합니다. (예: 'Aspirin')
    if not isinstance(args, dict):
        return {"drug_name": str(args)}

    # 2. 'drug_name' 키를 표준적으로 확인합니다.
    if "drug_name" in args and isinstance(args["drug_name"], str):
        return args

    # 3. 알려진 별칭의 포괄적인 목록을 확인합니다.
    known_aliases = ["drug", "name", "query", "medication", "medications", "drug_names", "drugs"]
    for alias in known_aliases:
        if alias in args:
            value = args[alias]
            if isinstance(value, str):
                return {"drug_name": value}
            if isinstance(value, list) and value:
                # 목록의 첫 번째 항목을 우선적으로 처리합니다.
                return {"drug_name": str(value[0])}

    # 4. 단일 키 딕셔너리 또는 깊게 중첩된 구조를 처리하기 위한 고급 대체 로직
    for key, val in args.items():
        if isinstance(val, str):
            # {"option_a": "Aspirin"}와 같은 경우를 처리합니다.
            return {"drug_name": val}
        if isinstance(val, dict):
            # 중첩된 딕셔너리를 위해 adapt_args를 재귀적으로 호출합니다.
            recursive_result = adapt_args(name, val)
            if "drug_name" in recursive_result and recursive_result["drug_name"]:
                return recursive_result

    # 최종 대체 로직: 유효한 약물 이름이 발견되지 않은 경우
    return {"drug_name": ""}

def _execute_tool_call(tool_name: str, args: Dict[str, Any], allowed_tools: set, tool_universe: ToolUniverse):
    """Handles tool validation and execution."""
    if tool_name not in allowed_tools: # LLM이 허용되지 않은 도구를 호출한 경우
        return {"error": f"Tool '{tool_name}' not allowed.", "allowed": sorted(list(allowed_tools))}
    
    if args is None:
        return {"error": "Invalid JSON arguments. Check your formatting."}

    try:
        fixed_args = adapt_args(tool_name, args)
        result = cached_run(tool_universe, tool_name, **fixed_args) #Tool 실핼
        if result is None:
            return {"error": f"Tool '{tool_name}' returned no result."}
        return result
    except Exception as e:
        return {"error": f"An error occurred while running the tool: {str(e)}"}

def suggest_tool_by_question(q: str) -> str | None:
    """
    질문의 키워드를 기반으로 도구를 제안합니다.
    다양한 키워드와 복합적인 질문에 대응할 수 있도록 로직을 확장했습니다.
    """
    s = (q or "").lower()

    # 상호작용 관련 키워드
    if any(k in s for k in ["interact", "상호작용", "병용", "함께 복용", "DDI"]):
        return "FDA_get_drug_interactions_by_drug_name"

    # 금기사항 관련 키워드
    if any(k in s for k in ["contraindication", "금기", "복용하면 안되는", "피해야 할"]):
        return "FDA_get_contraindications_by_drug_name"

    # 임신 관련 키워드
    if any(k in s for k in ["pregnan", "임신", "임산부", "태아"]):
        return "FDA_get_pregnancy_effects_info_by_drug_name"

    # 수유 관련 키워드
    if any(k in s for k in ["breastfeed", "수유", "모유", "lactation", "lactat"]):
        return "FDA_get_pregnancy_or_breastfeeding_info_by_drug_name"

    # 경고/위험/주의사항 관련 키워드
    if any(k in s for k in ["warning", "주의", "위험", "risk", "부작용", "adverse"]):
        return "FDA_get_risk_info_by_drug_name"

    # 용량 조절 관련 키워드 (새로운 기능)
    if any(k in s for k in ["dose", "용량", "조절", "신장", "renal", "간", "hepatic"]):
        # 복합적인 질문에 대응하기 위해 신장/간 관련 키워드에 따라 우선순위 결정
        if any(k in s for k in ["신장", "renal"]):
            return "renal_dose_adjust"
        if any(k in s for k in ["간", "hepatic"]):
            return "hepatic_dose_adjust"
        return "fda_label_lookup" # 일반적인 용량 정보 조회 도구로 대체

    # 약물 제형/형태 관련 키워드 (새로운 기능)
    if any(k in s for k in ["form", "제형", "형태", "캡슐", "정제", "시럽"]):
        return "fda_label_lookup" # 라벨 정보를 통해 제형 정보를 얻는 도구로 연결

    # 일반 약물 정보/적응증 관련 키워드 (새로운 기능)
    if any(k in s for k in ["info", "정보", "작용", "효과", "적응증", "diagnosis", "기전"]):
        return "fda_label_lookup"

    return None

def run_agent(question: str, options: Dict[str, str], max_rounds=6, model="gpt-4o-mini"):
    """
    Executes a multi-turn agent loop to answer a question using tools.
    """
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"Question:\n{question}\n\nOptions:\n{json.dumps(options, ensure_ascii=False)}"}
    ]
    trace: List[Dict[str, Any]] = []
    suggested_tool = suggest_tool_by_question(question)
    tool_called_in_round = False

    for round_num in range(max_rounds):
        # 1. Ask the model for its next action
        out = _ask(messages, model=model)
        
        # 2. Extract tool call from the response
        tool_name, tool_args = _extract_call(out)
        canonical_name = canonicalize_tool_name(tool_name) if tool_name else None

        # 3. Handle tool calls
        if canonical_name:
            tool_called_in_round = True
            tool_result = _execute_tool_call(canonical_name, tool_args, ALLOW_TOOLS, tu)
            
            trace.append({
                "tool_name": tool_name,
                "canonical_name": canonical_name,
                "args": tool_args,
                "result": tool_result
            })
            
            # Append model's thought process and the tool result to the conversation history
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": f"[{canonical_name} RESULT]\n{json.dumps(tool_result, ensure_ascii=False)}\nUse this information to continue."})
            
            # Continue to the next round if a tool was called
            continue
        
        # 4. If no tool was called, check for a final answer
        messages.append({"role": "assistant", "content": out})
        
        m = re.findall(r"Final answer:\s*([A-D])\b", out, flags=re.I)
        if m:
            return {"final_choice": m[-1].upper(), "rationale": out, "tools": trace}
        
        # 5. If it's the first round and no tool was called, or if tools failed, give a hint
        if round_num == 0 and not tool_called_in_round and suggested_tool:
            examples = []
            for k, v in list(options.items())[:2]:
                drug_name = v.split(';')[0].strip().split(',')[0].strip()
                examples.append(f'CALL {suggested_tool} {{"drug_name": "{drug_name}"}}')
            hint = f"Suggestion: Call the relevant tool for each option. For example:\n" + "\n".join(examples)
            messages.append({"role": "user", "content": hint})
            continue
        
        # 6. Fallback mechanism: If we reach this point, nudge the model to provide a final answer
        messages.append({"role": "user", "content": "You did not call a tool or provide a final answer. Based on the information you have, output EXACTLY one line in the format: Final answer: X (where X is A/B/C/D). No other text."})
        forced_out = _ask(messages, model=model, temperature=0).strip()
        
        m = re.findall(r"\s*([A-D])\b", forced_out, flags=re.I)
        letter = (m[-1].upper() if m else "")
        if letter:
            return {"final_choice": letter, "rationale": out, "tools": trace}
    
    # 7. Max rounds reached without a clear answer
    return {"final_choice": "", "rationale": "Max rounds reached without a final answer.", "tools": trace}