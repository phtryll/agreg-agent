import numpy as np
from typing import Dict, List
from sentence_transformers import CrossEncoder
from source.state import AgentState


# Build the vector DB and define a search function for the retriever
class EmbeddingIndex:
    def __init__(self, model_name: str, chunks: List[str]) -> None:
        self.model = CrossEncoder(model_name)
        self.chunks = chunks


    def search(self, query: str, k: int) -> List[str]:
        scores = self.model.predict([(query, chunk) for chunk in self.chunks])
        idx = np.argsort(scores)[-k:][::-1]
        return [self.chunks[i] for i in idx]


# Retrieve passages call
def retrieve_passages(
        state: AgentState,
        index: EmbeddingIndex,
        n_results: int = 3
    ) -> AgentState:
    questions = state.get("queries", [])
    if not questions:
        state["issues"].append("retriever: no questions available")
        state["retrieved"] = {}
        return state

    retrieved: Dict[str, List[str]] = {}
    for q in questions :
        retrieved[q] = index.search(q, n_results)

    state["retrieved"] = retrieved
    return state