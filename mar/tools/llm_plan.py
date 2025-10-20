# tools/llm_plan_tool.py
import re
import json
import requests
import logging
from string import Template
from typing import Dict, Any, List
from mar.tools.query_analysis import QueryAnalysisTool

log = logging.getLogger(__name__)

class LLMPlanTool:
    def __init__(self, cfg: Dict, query_analysis_tool: QueryAnalysisTool):
        self.cfg = cfg
        self.session = requests.Session()
        self.query_analysis_tool = query_analysis_tool
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self):
        return Template("""
You are a Retrieval Strategy Planner, a world-class expert in Information Retrieval. Your goal is to devise the optimal, multi-stage retrieval plan for a given query on the dataset. You must synthesize three sources of information:

1.  Your expert deconstruction of the query text.
2.  The detailed empirical performance data of available retrievers (Core Context).
3.  Statistical guidance from a dynamic query analysis tool.

---

#### 1. STRATEGIC GUIDANCE (FROM QUERY ANALYSIS TOOL)

This context provides a strong statistical prior about the query's likely dataset and top-performing retrievers. think deeper than just using the highest retriever.

- Analyze the performance of retrievers on the dataset, if the lexical retrievers like bm25 or splade outperformes th rest, then this dataset might be 
    argumentative dataset, in which longer documents are favorable. so consider building multiple or single retrieval plan from lexical.
- Analyze the info about the dataset, some dataset needs sementic anaylsis some lexical, and some needs both for high recall and high percision.
- Some dataset where neural retriever like bge shows superiority, so maybe consider enrich it with lexical or another semantic retriever in a multiple stage format to boost NDCG@10 .
- Usually if all retrievers have relatively high perceion, then a good plan is to use as many good retrievers with high recall in the initial stage, and in the later stage use a high perecion retriever.
- Weigh the benefits of adding a second stage. sometimes adding bge for the second stage is not beneficial.
- some query are simple and does not require a multiple stage retrieval plan, so maybe consider using the best performing retriever on its dataset.
- The imprtnant thing is you should think, and think deeper before building the retrieval plan.
                        
$analysis_context

                        
#### 2. Available retriever methods:
    -bm25: simple lexical retriever.
    -contriever: medium dense retriever.
    -splade: learned lexical retriever.
    -bge: large dense retriever.


#### 3. TASK & OUTPUT (STRICT JSON)

For the given query, perform the thought process below and then provide the final plan in a **single JSON block**.

**CRITICAL RULE:** For any stage with `stage_id > 1`, you MUST include an `"inputs"` key referencing the previous stage's output, like `"inputs": ["stage_1_output"]`. This links the stages into a pipeline.

---
#### EXAMPLES

Plan:
                        
{                
 "plan": [
    {
      "stage_id": 1, "action": "retrieve", "agents": ["splade"],
      "description": "state your reason for the retrievers choice"
    },                       
}

or 
                                     
{
  "plan": [
    {
      "stage_id": 1, "action": "retrieve", "agents": ["bm25"],
      "description": "argumentative query, so i need to retrieve long documents"
    },
    {
      "stage_id": 2, "action": "retrieve", "agents": ["splade"],
      "inputs": ["stage_1_output"],
      "description": "refining the retrieved douments from the first stage"
    }
  ]
}
or 
                                     
{
  "plan": [
    {
      "stage_id": 1, "action": "retrieve", "agents": ["bm25,splade,contriever"],
      "description": "argumentative query, so i need to retrieve long documents"
    },
    {
      "stage_id": 2, "action": "retrieve", "agents": ["bge"],
      "inputs": ["stage_1_output"],
      "description": "refining the retrieved douments from the first stage"
    }
  ]
}
""")
    
    @staticmethod
    def _format_tool_output_for_prompt(tool_output: Dict) -> str:
        if not tool_output or "dataset_stats" not in tool_output:
            return "No analysis context available."
        
        dataset = tool_output['predicted_dataset']
        confidence = tool_output['confidence']
        stats = tool_output['dataset_stats']
        
        if "error" in stats:
            return f"Analysis Error: {stats['error']}"

        best_precision = stats['best_retrievers']['by_precision_nDCG_at_10']
        best_recall = stats['best_retrievers']['by_recall_R_at_100']
        
        return (
            f"- Predicted Dataset: {dataset} (Confidence: {confidence:.1%})\n"
            f"- Dataset Domain: {stats['metadata'].get('domain', 'N/A')}\n"
            f"- Best Retriever for Precision (nDCG@10): '{best_precision['name']}' (Score: {best_precision['score']})\n"
            f"- Best Retriever for Recall (R@100): '{best_recall['name']}' (Score: {best_recall['score']})"
        )

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
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.RequestException as e:
            log.error(f"LLM Planner API call failed: {e}")
            return ""

    @staticmethod
    def _clean_json_string(s: str) -> str:
        s = s.strip().replace('“','"').replace('”','"').replace('‘',"'").replace('’',"'").replace('—','-').replace('–','-')
        s = s.replace('{{','{').replace('}}','}')
        s = re.sub(r',(\s*[}\]])', r'\1', s)
        return s

    @staticmethod
    def _extract_json_candidates(text: str) -> List[str]:
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

    @staticmethod
    def _validate_plan_structure(data: Any) -> bool:
        if not isinstance(data, dict) or "plan" not in data or not isinstance(data["plan"], list) or not data["plan"]:
            return False
        for i, stage in enumerate(data["plan"]):
            if not all(k in stage for k in ["stage_id", "agents"]) or not isinstance(stage["agents"], list) or not stage["agents"]:
                return False
        return True

    def _parse_plan(self, response_text: str) -> Dict:
        log.debug(f"RAW LLM PLAN RESPONSE:\n{response_text}")
        for cand in self._extract_json_candidates(response_text):
            try:
                parsed = json.loads(self._clean_json_string(cand))
                if self._validate_plan_structure(parsed):
                    log.info("Plan parsed successfully.")
                    return parsed
            except json.JSONDecodeError:
                continue
        log.warning("All parsing attempts failed. Using default single-stage plan (bge).")
        return self._get_default_plan()

    @staticmethod
    def _get_default_plan() -> Dict:
        return {"plan": [{"stage_id": 1, "action": "retrieve", "agents": ["bge"]}]}

    
    def run(self, query: str, query_meta: Dict) -> Dict:
        print("    L--> 🛠️  **Tool: LLMPlanTool**")
        print("        |  **Action:** Generating a multi-stage retrieval plan.")
        print("        |  **Step 1:** Analyzing query with QueryAnalysisTool...")
        tool_input = {"query": query, "entities": query_meta.get("entities", [])}
        tool_output = self.query_analysis_tool.run(tool_input)
        analysis_context = self._format_tool_output_for_prompt(tool_output)
        print(f"        |  **Context:** {analysis_context}")
        print("        |  **Step 2:** Calling LLM with formatted prompt...")
        prompt = self.prompt_template.safe_substitute(query=query, analysis_context=analysis_context)
        plan = self._call_api(prompt)
        print("        |  **Outcome:** Plan generated successfully.")
        print("   L--> ✅ **Tool Complete**\n")
        return self._parse_plan(plan)