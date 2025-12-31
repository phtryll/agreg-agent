from typing import Dict
from pydantic import BaseModel
from source.state import AgentState
from source.ollama import call_ollama


class TranscribeOutput(BaseModel):
    ipa: Dict[str, str]


# LLM call for this node
def transcribe(state: AgentState, model: str) -> AgentState:
    lexical_items = state.get("lexical_items", [])

    if not lexical_items:
        state["issues"].append("transcribe: lexical items empty")
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
        "Transcris en alphabet phonétique international (API) les lexèmes ci-dessous.\n"
        "Utilise des crochets [ ].\n"
        "Donne UNE forme phonétique par lexème.\n"
        "N'ajoute aucun commentaire.\n\n"
        "Lexèmes à transcrire :\n"
        + "\n".join(f"- {item}" for item in lexical_items)
    )

    try:
        parsed = call_ollama(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=TranscribeOutput,
        )
        state["ipa"] = parsed.ipa

    except Exception:
        state["issues"].append("transcribe: schema validation failed")

    return state