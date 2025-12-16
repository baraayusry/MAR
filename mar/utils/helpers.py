import sys

import json
import re
from typing import List, Dict
from mar.config import AGENT_ALIASES
from mar.utils.logging import get_logger

def normalize_agents(agent_names: List[str]) -> List[str]:
    log = get_logger()
    normed = []
    for a in agent_names:
        key = AGENT_ALIASES.get(a.strip().lower())
        if key is None:
            log.warning(f"Unknown agent in plan: '{a}'. Known aliases: {sorted(AGENT_ALIASES.keys())}")
            continue
        normed.append(key)
    seen = set()
    out = []
    for x in normed:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def merge_results(results_list: List[Dict[str, float]]) -> Dict[str, float]:
    merged = {}
    for results in results_list:
        if not results: continue
        for doc_id, score in results.items():
            if doc_id not in merged or score > merged[doc_id]:
                merged[doc_id] = score
    return merged

def pretty_plan(plan: Dict) -> str:
    try:
        return json.dumps(plan, indent=2, ensure_ascii=False)
    except Exception:
        return str(plan)

def clean_json_string(s: str) -> str:
    s = s.strip().replace('“','"').replace('”','"').replace('‘',"'").replace('’',"'").replace('—','-').replace('–','-')
    s = s.replace('{{','{').replace('}}','}')
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    return s

def extract_json_candidates(text: str) -> List[str]:
    candidates = []
    candidates += re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    candidates += re.findall(r"Plan:\s*(\{[\s\S]*?\})", text, flags=re.IGNORECASE)
    brace_depth, start_idx = 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0: start_idx = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                candidates.append(text[start_idx:i+1])
    seen = set()
    return [c for c in candidates if c not in seen and not seen.add(c)]