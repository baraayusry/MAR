# agents/planner_agent.py
from typing import Dict, Any

class PlannerAgent:
    def __init__(self, plan_tool: Any):
        self.plan_tool = plan_tool

    def run(self, query: str, context: Dict[str, Any]) -> Dict:
        print("🤖 **Agent: PlannerAgent**")
        print("   |  **State:** Analyzing request...")
        if context.get("plan"):
            print("   |  **Decision:** A pre-defined plan was provided.")
            print("   |  **Action:** Using the existing plan.")
            print("🤖 **Agent Complete**\n")
            return context["plan"]
        else:
            print("   |  **Decision:** No plan found in the context.")
            print("   |  **Action:** Delegating to LLMPlanTool to generate a new one.")
            query_meta = context.get('meta', {})
            plan = self.plan_tool.run(query, query_meta)
            print("🤖 **Agent Complete**\n")
            return plan