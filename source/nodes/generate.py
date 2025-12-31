from typing import Optional, List, Annotated
from pydantic import BaseModel, Field
from source.state import AgentState
from source.ollama import call_ollama


# Practical typing placeholders
NonEmptyStrList = Annotated[List[str], Field(min_length=1)]
NonEmptyStageList = Annotated[List["Stage"], Field(min_length=1)]


# There are multiple stages for one derivation
class Stage(BaseModel):
    period: str = Field(..., min_length=1)
    forms: NonEmptyStrList
    changes: NonEmptyStrList


# Full structured generation output
class Derivation(BaseModel):
    latin_lemma: str = Field(..., min_length=1)
    latin_ipa: str = Field(..., min_length=1)
    modern_lemma: str = Field(..., min_length=1)
    modern_ipa: str = Field(..., min_length=1)
    stages: NonEmptyStageList
    remarks: Optional[List[str]] = None


# LLM call for this node
def generate(state: AgentState, model: str) -> AgentState:
    source_lemma = state["lexical_items"][0]
    target_lemma = state["lexical_items"][1]
    source_ipa = state["ipa"].get(source_lemma, "")
    target_ipa = state["ipa"].get(target_lemma, "")

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
        "Produis une dérivation phonétique structurée en respectant le schéma fourni.\n\n"
        "Données fournies (à réutiliser telles quelles, sans les modifier) :\n"
        f"- Lemme latin : {source_lemma}\n"
        f"- API latin : {source_ipa}\n"
        f"- Lemme français moderne : {target_lemma}\n"
        f"- API français moderne : {target_ipa}\n\n"
        "Étapes chronologiques à inclure (ne rien ajouter, ne rien supprimer) :\n"
        + "\n".join(f"- {step}" for step in state["plan"])
        + "\n\n"
        "Consignes :\n"
        "- utiliser l'API latine pour la première étape\n"
        "- utiliser l'API française moderne pour la dernière étape\n"
        "- pour les étapes intermédiaires, inférer la forme API suivant les évolutions\n"
    )

    try:
        derivation = call_ollama(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=Derivation,
        )
        state["structured_answer"] = derivation

    except Exception:
        state["issues"].append("generate: structured derivation validation failed")

    return state