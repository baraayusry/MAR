import concurrent.futures
from typing import Dict, List, Any, Optional

def concatenate_and_deduplicate(results_list: List[Dict[str, float]]) -> Dict[str, float]:
    unique_doc_ids = set()
    for results in results_list:
        if results:
            unique_doc_ids.update(results.keys())
    return {doc_id: 1.0 for doc_id in unique_doc_ids}

class RetrieveTool:
    def __init__(self, retrievers: Dict[str, Any]):
        self.retrievers = retrievers

    def run(self, query: str, algorithms: List[str], top_k: int, candidate_docs: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        print("    L--> 🛠️  **Tool: RetrieveTool**")
        if candidate_docs:
            print(f"        |  **Action:** Performing a REFINEMENT search on {len(candidate_docs)} docs.")
        else:
            print("        |  **Action:** Performing INITIAL search on the full corpus.")
        print(f"        |  **Algorithms:** {', '.join(algorithms)}")

        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(algorithms)) as executor:
            future_to_retriever = {}
            for name in algorithms:
                if name not in self.retrievers:
                    print(f"        |  **Warning:** Retriever '{name}' not found. Skipping.")
                    continue

                retriever = self.retrievers[name]
                if candidate_docs:
                    if name == 'bm25':
                        print("        |  **Info:** Skipping bm25 for refinement search.")
                        continue
                    future = executor.submit(retriever.search_subset, query, candidate_docs, top_k)
                else:
                    future = executor.submit(retriever.search, query, {}, top_k)
                future_to_retriever[future] = name

            for future in concurrent.futures.as_completed(future_to_retriever):
                retriever_name = future_to_retriever[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as exc:
                    print(f"        |  **Error:** Retriever '{retriever_name}' failed: {exc}")

        
        final_results = concatenate_and_deduplicate(all_results)

        print(f"        |  **Outcome:** Concatenated results into {len(final_results)} unique documents.")
        print("   L--> ✅ **Tool Complete**\n")
        return final_results