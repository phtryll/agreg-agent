from typing import List, Dict
from pydantic import BaseModel
from source.state import AgentState
from source.ollama import call_ollama


class TriggersOutput(BaseModel):
    triggers: List[str]


# LLM call for this node
def infer_triggers(state: AgentState, model: str) -> AgentState:
    lexical_items = state.get("lexical_items", [])
    if len(lexical_items) != 2:
        state["issues"].append("triggers: expected exactly 2 lexical items")
        state["triggers"] = []
        return state

    latin_lemma = lexical_items[0]
    french_lemma = lexical_items[1]
    ipa: Dict[str, str] = state.get("ipa", {})
    latin_ipa = ipa.get(latin_lemma)
    french_ipa = ipa.get(french_lemma)

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
        "Infère des évolutions phonétiques plausibles à partir de la comparaison de deux formes.\n\n"
        "Définition:\n"
        "Un évolution phonétique est un processus ou un type de transformation phonétique\n"
        "qui permet de rendre compte du passage d'une forme phonétique à une autre.\n\n"
        "Tu dois comparer les formes fournies et identifier les évolutions nécessaires\n"
        "pour passer de la forme latine à la forme française.\n\n"
        "Règles:\n"
        "- Utiliser exclusivement la terminologie standard de la phonétique historique du français\n"
        "- Nommer des évolutions (par exemple: amuïssement, nasalisation, diphtongaison, etc.)\n"
        "- Ne pas expliquer les évolutions\n"
        "- Ne pas fournir de chronologie\n"
        "- Ne pas justifier les réponses\n"
        "- Ne pas reformuler les formes\n\n"
        "Les évolutions doivent être compatibles avec les formes phonétiques données.\n\n"
        f"Forme latine : {latin_lemma}\n"
        f"API latin : {latin_ipa}\n"
        f"Forme française : {french_lemma}\n"
        f"API français : {french_ipa}\n"
    )

    try:
        parsed = call_ollama(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=TriggersOutput,
        )
        state["triggers"] = parsed.triggers
    except Exception:
        state["issues"].append("triggers: schema validation failed")
        state["triggers"] = []

    return state