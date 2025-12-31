from pydantic import BaseModel
from source.state import AgentState
from source.ollama import call_ollama


# Classify output
class ClassifyOutput(BaseModel):
    source_lemma: str
    target_lemma: str


# LLM call for this node
def classify(state: AgentState, model: str) -> AgentState:
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
        "Extrais UNIQUEMENT le lexème de départ (latin) et le lexème d'arrivée "
        "(français moderne) de l'évolution phonétique décrite par la question ci-dessous.\n"
        "L'évolution est généralement indiquée par le symbole '>'.\n\n"
        f"Question :\n{state['question']}\n"
    )

    try:
        parsed = call_ollama(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=ClassifyOutput,
        )
        state["lexical_items"] = [parsed.source_lemma, parsed.target_lemma]

    except Exception:
        state["issues"].append("classify: schema validation failed")

    return state