import re, json, requests, logging
from string import Template
from typing import Dict

class Planner:
    def __init__(self, cfg: Dict, logger: logging.Logger=None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger('Planner')
        self.session = requests.Session()

    @staticmethod
    def prompt_template() -> str:
        return '''
    You are an expert AI planning agent for a multi-retriever pipeline. Your job is to choose the best one- or two-stage retrieval plan for the query.

    # Available Retrieval Algorithms (ALL are retrievers)
    - bm25: Fast lexical, exact terms, quoted phrases, units, numbers.
    - contriever: Dense semantic for broad/ambiguous intents, paraphrases.
    - splade: Sparse neural for jargon/abbreviations/synonyms-heavy text.
    - bge: Strong dense retriever for precise semantic refinement (often Stage2).

    # Very Important
    - Do NOT include any "(filtered)" or extra suffixes; the agent decides filtering later.
    - Use EXACT tokens: bm25, contriever, splade, bge.
    - Output JSON ONLY for the plan line, no extra prose in that line.
    - Use " + " to join algorithms in Stage1 or Stage2.
    - Keep to at most 3 algorithms in a single stage.

    # How to think (internal, do NOT reveal)
    <scratchpad>
    1) Identify query signals:
    - Exact phrase? quotes? numbers/units? => bm25 (+ splade if synonyms/jargon)
    - Heavy jargon/abbreviations? domain-specific? => splade (+ contriever)
    - Broad/ambiguous topic or long paraphrase? => contriever (+ bm25 or splade)
    - Need strong semantic refinement? => add Stage2: bge
    - Very short or underspecified query? => bm25 + contriever; consider Stage2 bge
    2) Choose one stage if simple and precise; two stages if broad/ambiguous and benefits from dense refinement.
    3) Prefer bm25 for high-precision lexical anchors; combine with splade for synonymy/jargon; include contriever for breadth.
    4) Use bge typically as Stage2 only when a second stage is warranted.
    </scratchpad>

    # Output format
    Return:
    Query: "{query}"
    Rationale: one short sentence (<= 20 words).
    Plan: {{"Stage1": "...", "Stage2": "..."}}    # Stage2 optional

    # Examples

    Query: "What is the specific heat capacity of liquid water?"
    Rationale: Exact factual keyworded query; lexical is sufficient.
    Plan: {{"Stage1": "bm25"}}

    ---
    Query: "Find clinical studies on the efficacy of CAR-T therapy for non-Hodgkin lymphoma"
    Rationale: Medical jargon and variants; combine sparse and dense.
    Plan: {{"Stage1": "splade + contriever"}}

    ---
    Query: "research on the psychological effects of social media"
    Rationale: Broad topic; start wide then refine semantically.
    Plan: {{"Stage1": "bm25 + contriever + splade", "Stage2": "bge"}}

    ---
    Query: "SARS-CoV-2 spike glycoprotein conformational dynamics"
    Rationale: Domain jargon; sparse + dense capture terminology and meaning.
    Plan: {{"Stage1": "splade + contriever"}}

    ---
    Query: "heart attack early symptoms" 
    Rationale: Synonyms/paraphrases (MI vs heart attack); lexical plus semantic.
    Plan: {{"Stage1": "bm25 + contriever"}}

    ---
    Query: "CRISPR off-target detection methods"
    Rationale: Technical terminology with variants; add semantic refinement.
    Plan: {{"Stage1": "splade + bm25", "Stage2": "bge"}}

    ---
    Query: "GDP growth of Japan 2010–2015"
    Rationale: Numeric/time-bounded; lexical anchors suffice.
    Plan: {{"Stage1": "bm25"}}

    ---
    # Task
    Follow the thinking process silently, then produce the final lines below. Do not include the scratchpad.

    Query: "{query}"
    Rationale:
    Plan:
    '''



    def _call(self, prompt: str) -> str:
        url = f"{self.cfg['base_url']}/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg['api_key']}", "Content-Type": "application/json"}
        payload = {"model": self.cfg['model'], "messages": [{"role":"user","content": prompt}], "temperature": self.cfg.get('temperature',0.0), "max_tokens": self.cfg.get('max_tokens',512)}
        try:
            r = self.session.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            self.logger.error(f"Planner LLM failed: {e}")
            return ""

    def plan(self, query: str) -> Dict:
        prompt = self.prompt_template(query=query).format(query=query)
        raw = self._call(prompt)
        m = re.search(r"\{[\s\S]*\}", raw or "")
        if not m:
            self.logger.warning("Planner returned no JSON; using default plan")
            return self.cfg.get('default_plan', {"Stage1": "bm25 + contriever"})
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except Exception:
            self.logger.warning("Planner JSON parse failed; using default plan")
            return self.cfg.get('default_plan', {"Stage1": "bm25 + contriever"})

