import requests
import json
from string import Template
from typing import Dict, List, Any
from mar.utils.helpers import  clean_json_string, extract_json_candidates
from mar.utils.logging import get_logger
from mar.tools.query_analysis import QueryAnalysisTool

log = get_logger()

class PlannerAgent:
    def __init__(self, cfg: Dict, tools: List[Any], prompt: str = ""):
        self.cfg = cfg
        self.tools = tools
        self.session = requests.Session()
        
        if prompt:
            self.prompt_template = Template(prompt)
        else:
            self.prompt_template = self._load_default_prompt_template()

    def _load_default_prompt_template(self):
        return Template("""
You are a Retrieval Strategy Planner, a world-class expert in Information Retrieval. Your goal is to devise the optimal, multi-stage retrieval plan.

$analysis_context

#### TASK & OUTPUT (STRICT JSON)
For the given query, perform the thought process and provide the final plan in a single JSON block.
Plan:
{ "plan": [ { "stage_id": 1, "action": "retrieve", "agents": ["splade"] } ] }
""")

    def _get_analysis_tool(self) -> QueryAnalysisTool:
        for tool in self.tools:
            if isinstance(tool, QueryAnalysisTool):
                return tool
        return None

    def _format_tool_output(self, tool_output: Dict) -> str:
        if not tool_output or "dataset_stats" not in tool_output: return "No analysis."
        stats = tool_output['dataset_stats']
        if "error" in stats: return f"Error: {stats['error']}"
        best_p = stats['best_retrievers']['by_precision_nDCG_at_10']
        return f"- Predicted Dataset: {tool_output['predicted_dataset']} (Conf: {tool_output['confidence']:.1%})\n- Best Precision Retriever: {best_p['name']}"

    def _call_api(self, prompt: str) -> str:
        payload = {
            "model": self.cfg['model'],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.get('temperature', 0.0),
            "max_tokens": self.cfg.get('max_tokens', 1024)
        }
        url = f"{self.cfg['base_url']}/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg['api_key']}", "Content-Type": "application/json"}
        try:
            with self.session.post(url, headers=headers, json=payload, timeout=60) as response:
                response.raise_for_status()
                return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            log.error(f"Planner API failed: {e}")
            return ""

    def generate_plan(self, query: str, query_meta: Dict) -> Dict:
        tool = self._get_analysis_tool()
        analysis_context = "No query analysis tool provided."
        
        if tool:
            tool_output = tool.run({"query": query, "entities": query_meta.get("entities", [])})
            analysis_context = self._format_tool_output(tool_output)
            
        log.info(f"--- PLANNER CONTEXT ---\n{analysis_context}")
        
        prompt = self.prompt_template.safe_substitute(query=query, analysis_context=analysis_context)
        response = self._call_api(prompt)
        
        for cand in extract_json_candidates(response):
            try:
                parsed = json.loads(clean_json_string(cand))
                if isinstance(parsed, dict) and "plan" in parsed: return parsed
            except: continue
        
        # Default fallback plan
        return {"plan": [{"stage_id": 1, "action": "retrieve", "agents": ["bge"]}]}