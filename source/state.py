from typing import TypedDict, List, Dict, Optional, Any


class AgentState(TypedDict):
    # The question
    question: str

    # Identified lexical items (classify node)
    lexical_items: List[str]

    # IPA transcription for lexical items (transcribe node)
    ipa: Dict[str, str]

    # Hypothesis/inference of phonetic evolutions (triggers node)
    triggers: List[str]

    # Queries for the retriever (queries node)
    queries: List[str]

    # Retrieved passages (retriever node)
    retrieved: Dict[str, List[Any]]

    # The plan based on the 'phonological facts' from the retrieved chunks (planner node)
    plan: List[Dict[str, Any]]

    # Structured derivation produced by generate (generate node)
    structured_answer: Optional[Any]

    # Final rendered answer
    answer: str

    # Issues detected during processing
    issues: List[str]