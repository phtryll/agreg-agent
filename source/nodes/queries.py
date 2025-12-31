from typing import List
from pydantic import BaseModel
from source.state import AgentState
from source.ollama import call_ollama


class QuestionsOutput(BaseModel):
    questions: List[str]


# LLM call for this node
def infer_phonological_questions(state: AgentState, model: str) -> AgentState:
    triggers = state.get("triggers", [])

    if not triggers:
        state["issues"].append("questions: no triggers available")
        state["queries"] = []
        return state

    system_prompt: str = (
        "Tu es un candidat à l'agrégation de lettres modernes. "
        "Tu passes l'épreuve d'étude grammaticale d'un texte antérieur à 1500, "
        "et tu dois répondre à une question de phonétique historique.\n"
        "Tu dois produire UNIQUEMENT du JSON valide conforme exactement au schéma fourni.\n"
        "Aucune explication supplémentaire. Aucune clé supplémentaire.\n"
        "Question (contexte): "
        f"{state['question']}"
    )

    user_prompt: str = (
        "À partir des informations listés ci-dessous, "
        "formule des questions pertinentes de phonétique historique du français.\n\n"
        "Ces questions doivent permettre de rechercher, dans un manuel de phonétique historique, "
        "les informations nécessaires pour comprendre l'évolution phonétique d'un mot du latin "
        "au français.\n\n"
        "Informations a mettre sous forme de question:\n"
        + "\n".join(f"- {t}" for t in triggers)
    )

    try:
        parsed = call_ollama(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=QuestionsOutput,
        )
        state["queries"] = parsed.questions
    except Exception:
        state["issues"].append("questions: schema validation failed")
        state["queries"] = []

    return state