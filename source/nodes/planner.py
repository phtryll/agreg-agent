from typing import List, Dict, Any
from pydantic import BaseModel
from source.state import AgentState
from source.ollama import call_ollama


# Structured output format
class PhonologicalFact(BaseModel):
    phenomenon: str
    description: str
    temporal_marker: str | None
    graphie: str | None
    source_excerpt: str


# The planner is basically a list of "facts" that the generator has to take into account
class PlanOutput(BaseModel):
    facts: List[PhonologicalFact]


# LLM call for this node
def plan(state: AgentState, model: str) -> AgentState:
    retrieved: Dict[str, List[Any]] = state.get("retrieved", {})

    if not retrieved:
        state["issues"].append("planner pass 3: no retrieved passages")
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

    evidence_blocks: List[str] = []

    for question, passages in retrieved.items():
        block = (
            f"Question de recherche : {question}\n"
            "Extraits pertinents :\n"
            + "\n".join(f"- {p}" for p in passages)
        )
        evidence_blocks.append(block)

    user_prompt: str = (
        "À partir des extraits ci-dessous issus d'un manuel de phonétique historique, "
        "extrais des faits phonologiques explicitement attestés par le texte.\n\n"
        "Un fait phonologique correspond à un phénomène nommé de la phonétique historique, "
        "accompagné d'une brève description de ce qui se produit.\n"
        "Lorsque le manuel le précise explicitement, tu peux indiquer une information temporelle "
        "ou des informations particulières concernant la graphie.\n\n"
        "Tu ne dois pas produire de narration ni de chronologie complète, "
        "mais uniquement isoler des faits fondés sur les extraits fournis.\n\n"
        "Chaque fait doit obligatoirement être appuyé par un extrait du manuel recopié textuellement.\n"
        "N'infère rien qui ne soit pas explicitement présent dans les passages.\n"
        "N'introduis ni périodes, ni datations, ni exemples lexicaux qui ne figurent pas dans les extraits.\n"
        "Si tu n'as pas d'informations concernant un champ, marque 'aucune info trouvée'.\n\n"
        "Extraits du manuel :\n"
        + "\n\n".join(evidence_blocks)
    )

    try:
        parsed = call_ollama(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=PlanOutput,
        )
        state["plan"] = [fact.model_dump() for fact in parsed.facts]

    except Exception:
        state["issues"].append("planner: schema validation failed")
        state["plan"] = []

    return state
